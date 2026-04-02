"""
genesis_uninavid_batch_parallel.py

N 个 VLN episode 在同一 Genesis 场景中真正并行评估。

并行化策略：
  - Genesis BatchRenderer 一次 GPU call 渲染所有 N 个第一视角画面（主要加速点）
  - NavMesh 碰撞/滑动/落地检测全部向量化 tensor 运算
  - UniNaVid 模型权重只加载一份；per-agent 的 feat_cache / long_feat_cache
    维护在外部，推理前 swap in、推理后 swap out（顺序执行，共享 GPU 显存）
  - 用 active_mask 追踪提前终止的 agent

为什么不做真正的 batch=N 推理？
  model.get_model().feat_cache 是形状 (k, tokens, hidden) 的单一全局 tensor，
  vlm_attention 中有 assert final_token_nav.shape[0] == 1 的硬编码约束。
  如果把 N 个 agent 拼成 batch 一起 forward，所有 agent 的历史帧会混在同一个
  feat_cache 里，结果完全错误。
  正确做法：批量渲染 → 顺序推理（swap cache）。
"""

import sys
sys.path.append("/data1/home/clk/project/Genesis")

import genesis as gs
import torch
import torch.nn.functional as F
import numpy as np
import os
import shutil
import gzip
import json
import time
import cv2
import multiprocessing

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# UniNaVid imports
from uninavid.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    VIDEO_START_SPECIAL_TOKEN, VIDEO_END_SPECIAL_TOKEN,
    IMAGE_START_TOKEN, IMAGE_END_TOKEN,
    NAVIGATION_SPECIAL_TOKEN, IAMGE_SEPARATOR,
    NAVIGATION_IDENTIFIER,
)
from uninavid.conversation import conv_templates, SeparatorStyle

# ---------------------------------------------------------------------------
# File Paths & Constants
# ---------------------------------------------------------------------------
EPISODES_FILE = "/data1/home/clk/navid_ws/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
SCENE_DATASETS = "/data1/home/clk/project/genark/scene_datasets"
MODEL_PATH = "model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"

NUM_EPISODES  = 1839

# 每个场景最多并行几条 episode（None = 不限制）
# 300 路会 OOM，先从小值开始试，逐步调大
MAX_PARALLEL_PER_SCENE = 128

CAMERA_HEIGHT = 0.88
CAM_RES       = (640, 480)
FOV           = 64
LIGHT_SCALE   = 4.0
MAX_STEPS     = 500       # per-episode step limit

STEP_MOVE     = 0.25
STEP_TURN     = np.deg2rad(30.0)
ALLOW_SLIDING = True
MAX_STEP_HEIGHT = 0.5

ACTION_MAP = {0: "stop", 1: "move_forward", 2: "turn_left", 3: "turn_right"}

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data():
    print(f"Loading episodes from {EPISODES_FILE} ...")
    if EPISODES_FILE.endswith(".gz"):
        with gzip.open(EPISODES_FILE, "rt") as f:
            data = json.load(f)
    else:
        with open(EPISODES_FILE, "r") as f:
            data = json.load(f)
    episodes = data["episodes"]
    print(f"  Found {len(episodes)} episodes.")
    return episodes


# ---------------------------------------------------------------------------
# GPU NavMesh helpers  (unchanged from original)
# ---------------------------------------------------------------------------

def batch_cross_2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def batch_is_point_in_triangle(pt, v0, v1, v2):
    def sign(p1, p2, p3):
        return (p1[..., 0] - p3[..., 0]) * (p2[..., 1] - p3[..., 1]) - \
               (p2[..., 0] - p3[..., 0]) * (p1[..., 1] - p3[..., 1])

    d1 = sign(pt, v0, v1)
    d2 = sign(pt, v1, v2)
    d3 = sign(pt, v2, v0)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def batch_get_z_on_triangle(v0, v1, v2, x, y):
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = torch.cross(edge1, edge2, dim=-1)
    mask = torch.abs(normal[..., 2]) < 1e-6
    d_x = x - v0[..., 0]
    d_y = y - v0[..., 1]
    num = normal[..., 0] * d_x + normal[..., 1] * d_y
    den = normal[..., 2]
    den = torch.where(torch.abs(den) < 1e-6, torch.ones_like(den), den)
    z = v0[..., 2] - num / den
    z = torch.where(mask.expand_as(z), torch.full_like(z, float("inf")), z)
    return z


def batch_find_floor(pos_2d, current_floor_z,
                     tri_v0_2d, tri_v1_2d, tri_v2_2d,
                     tri_v0_3d, tri_v1_3d, tri_v2_3d):
    pt   = pos_2d.unsqueeze(1)
    v0_2 = tri_v0_2d.unsqueeze(0)
    v1_2 = tri_v1_2d.unsqueeze(0)
    v2_2 = tri_v2_2d.unsqueeze(0)
    inside = batch_is_point_in_triangle(pt, v0_2, v1_2, v2_2)

    v0_3 = tri_v0_3d.unsqueeze(0)
    v1_3 = tri_v1_3d.unsqueeze(0)
    v2_3 = tri_v2_3d.unsqueeze(0)
    z_vals = batch_get_z_on_triangle(v0_3, v1_3, v2_3, pt[..., 0], pt[..., 1])

    z_diff = torch.abs(z_vals - current_floor_z.unsqueeze(1))
    height_ok = z_diff < MAX_STEP_HEIGHT
    valid_cand = inside & height_ok

    z_diff_m = torch.where(valid_cand, z_diff, torch.full_like(z_diff, float("inf")))
    min_dist, best_idx = torch.min(z_diff_m, dim=1)
    valid_mask = min_dist != float("inf")
    best_z = torch.gather(z_vals, 1, best_idx.unsqueeze(1)).squeeze(1)
    return valid_mask, best_z, best_idx


def batch_get_sliding_position(current_pos, desired_pos, current_tri_idx,
                                tri_v0_2d, tri_v1_2d, tri_v2_2d):
    p0 = tri_v0_2d[current_tri_idx]
    p1 = tri_v1_2d[current_tri_idx]
    p2 = tri_v2_2d[current_tri_idx]

    edge_starts = torch.stack([p0, p1, p2], dim=1)
    edge_ends   = torch.stack([p1, p2, p0], dim=1)
    edge_vecs   = edge_ends - edge_starts

    p_des = desired_pos.unsqueeze(1)
    p_cur = current_pos.unsqueeze(1)
    cp1 = batch_cross_2d(edge_vecs, p_des - edge_starts)
    cp0 = batch_cross_2d(edge_vecs, p_cur - edge_starts)
    crossing = (cp1 * cp0) < 0

    v_ap = p_des - edge_starts
    dot  = (v_ap * edge_vecs).sum(dim=-1)
    sq   = (edge_vecs * edge_vecs).sum(dim=-1)
    sq   = torch.where(sq < 1e-6, torch.ones_like(sq), sq)
    t    = torch.clamp(dot / sq, 0.0, 1.0)
    candidates = edge_starts + t.unsqueeze(-1) * edge_vecs

    has_crossing, first_idx = torch.max(crossing.long(), dim=1)
    selected = torch.gather(candidates, 1,
                            first_idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
    return torch.where(has_crossing.view(-1, 1).bool(), selected, current_pos)


def calculate_initial_yaw_tensor(rotations):
    """rotations: (N, 4) [x, y, z, w]"""
    x, y, z, w = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    dir_x_h = -2 * w * y
    dir_z_h = y * y - w * w
    return torch.atan2(-dir_z_h, dir_x_h)


# ---------------------------------------------------------------------------
# BatchUniNaVidWrapper
# ---------------------------------------------------------------------------

class BatchUniNaVidWrapper:
    """
    管理 N 个 agent 并行导航，共享一个 UniNaVid 模型实例。

    真正的 batch 推理流程（三阶段）：
      1. 批量视觉编码：把所有需推理 agent 的帧拼成一个大 batch，一次过
         vision_tower + mm_projector（比原先 N 次调用快 N 倍）。
      2. 逐 agent 更新 feat_cache + 构建 input_embeds：
         顺序执行，但只含 tensor concat / avg_pool / online_process_tensor，
         不含任何 LLM 前向传播，整体耗时很短。
      3. Batch LLM 生成：左 padding → attention_mask → 一次 model.generate()
         完成所有 agent 的自回归解码（batch=N，而非原先的 N×batch=1）。

    注意：
      UniNaVid 的 prepare_inputs_labels_for_multimodal 在 images=None 时会把
      inputs_embeds 覆盖为 None。因此 _batch_generate 会临时 patch model.forward，
      在"第一步使用预计算 embeds"时绕过图像编码路径，后续自回归步骤恢复正常。
    """

    PROMPT_TEMPLATE = (
        "Imagine you are a robot programmed for navigation tasks. "
        "You have been given a video of historical observations and an image "
        "of the current observation <image>. "
        "Your assigned task is: '{}'. "
        "Analyze this series of images to determine your next four actions. "
        "The predicted action should be one of the following: "
        "forward, left, right, or stop."
    )

    def __init__(self, model_path: str, n_agents: int):
        self.n_agents = n_agents
        self.conv_mode = "vicuna_v1"

        print(f"[BatchWrapper] Loading UniNaVid from {model_path} ...")
        model_name = get_model_name_from_path(model_path)
        (self.tokenizer, self.model,
         self.image_processor, self.context_len) = load_pretrained_model(
            model_path, None, model_name
        )
        self.model.eval()
        self.model.config.run_type = "eval"
        self._device = next(self.model.parameters()).device
        print(f"[BatchWrapper] Model loaded on device: {self._device}")

        _conv = conv_templates[self.conv_mode].copy()
        self.stop_str = (
            _conv.sep if _conv.sep_style != SeparatorStyle.TWO else _conv.sep2
        )
        self._special_ids = self._get_all_special_token_ids()

        # Per-agent state
        self.instructions:      list[str]       = [""] * n_agents
        self.rgb_lists:         list[list]      = [[] for _ in range(n_agents)]
        self.pending_actions:   list[list[int]] = [[] for _ in range(n_agents)]
        self.feat_caches:       list            = [None] * n_agents
        self.long_feat_caches:  list            = [None] * n_agents
        self.weights:           list[int]       = [1]    * n_agents

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_all(self, instructions: list[str]):
        """在所有 episode 开始前调用一次，重置全部 agent 状态。"""
        assert len(instructions) == self.n_agents
        for i in range(self.n_agents):
            self.instructions[i]     = instructions[i]
            self.rgb_lists[i]        = []
            self.pending_actions[i]  = []
            self.feat_caches[i]      = None
            self.long_feat_caches[i] = None
            self.weights[i]          = 1
        self.model.get_model().initialize_online_inference_nav_feat_cache()

    def step(self, rgb_batch_np: np.ndarray,
             active: np.ndarray, step_idx: int) -> list[int]:
        """
        每个 step 调用一次。

        阶段 1：消费缓存动作，收集本步需要推理的 agent 列表。
        阶段 2：Batch 视觉编码（一次 vision tower 前向传播）。
        阶段 3：逐 agent 更新 feat_cache + 手动构建 input_embeds。
        阶段 4：Batch LLM generate（一次 model.generate() 调用）。
        """
        actions = [0] * self.n_agents

        # ---- Phase 1: consume pending actions ----
        need_infer: list[int] = []
        for i in range(self.n_agents):
            if not active[i]:
                continue
            self.rgb_lists[i].append(rgb_batch_np[i].copy())
            if self.pending_actions[i]:
                actions[i] = self.pending_actions[i].pop(0)
            else:
                need_infer.append(i)

        if not need_infer:
            return actions

        # ---- Phase 2: preprocess & batch vision encoding ----
        video_tensors: list[torch.Tensor] = []
        frame_counts:  list[int]          = []
        valid_agents:  list[int]          = []
        for i in need_infer:
            vt = self._preprocess_frames(self.rgb_lists[i])
            if vt is not None:
                video_tensors.append(vt)
                frame_counts.append(vt.shape[0])
                valid_agents.append(i)
            else:
                actions[i] = 0

        if not video_tensors:
            return actions

        model_inner = self.model.get_model()
        nav_size  = self._get_nav_size()
        grid_size = self._get_grid_size()

        try:
            with torch.inference_mode():
                all_frames = torch.cat(video_tensors, dim=0)        # (ΣTi, C, H, W)
                all_raw = model_inner.get_vision_tower()(all_frames) # (ΣTi, P, Dv)
                if (self.model.config.mm_vision_select_feature == "patch"
                        and all_raw.shape[1] % 2 == 1):
                    all_raw = all_raw[:, 1:]                         # remove CLS token

                # ---- Phase 3: per-agent cache update + input_embeds ----
                input_embeds_list: list[torch.Tensor] = []
                offset = 0
                for agent_i, Ti in zip(valid_agents, frame_counts):
                    agent_raw = all_raw[offset: offset + Ti]         # (Ti, P, Dv)
                    offset += Ti

                    # Grid-compress: history (nav_size tok/frame) + current obs (64 tok)
                    vis_hist = self._process_grid(agent_raw, grid_size)  # (Ti, nav_size, Dv)
                    vis_cur  = self._process_grid(agent_raw[-1:], 8)     # (1,  64,       Dv)

                    # Project to LLM hidden size
                    vis_hist_proj = model_inner.mm_projector(vis_hist)   # (Ti, nav_size, Dllm)
                    vis_cur_proj  = model_inner.mm_projector(vis_cur)    # (1,  64,       Dllm)

                    # Swap in cache
                    model_inner.feat_cache      = self.feat_caches[agent_i]
                    model_inner.long_feat_cache = self.long_feat_caches[agent_i]
                    model_inner.weight          = self.weights[agent_i]
                    model_inner.new_frames      = Ti

                    # Append new history to feat_cache
                    old = model_inner.feat_cache
                    model_inner.feat_cache = (
                        torch.cat([old, vis_hist_proj], dim=0)
                        if old is not None else vis_hist_proj
                    )

                    # Compress history (updates long_feat_cache in-place)
                    compressed, lengths = self.model.online_process_tensor(nav_size)
                    # compressed: (L_hist, Dllm),  lengths: list[int]

                    # Build input_embeds for this agent (no LLM forward)
                    embeds_i = self._build_input_embeds(
                        agent_i, compressed, lengths, vis_cur_proj, model_inner
                    )
                    input_embeds_list.append(embeds_i)

                    # Swap out cache
                    self.feat_caches[agent_i]      = model_inner.feat_cache
                    self.long_feat_caches[agent_i] = model_inner.long_feat_cache
                    self.weights[agent_i]          = model_inner.weight
                    model_inner.feat_cache         = None
                    model_inner.long_feat_cache    = None
                    model_inner.weight             = 1
                    model_inner.new_frames         = 0

                    # Clear rgb buffer (frames now live in feat_cache)
                    self.rgb_lists[agent_i] = []

                # ---- Phase 4: one batched LLM generate call ----
                output_texts = self._batch_generate(input_embeds_list)

        except Exception as e:
            import traceback
            print(f"[BatchWrapper] step {step_idx} batch inference failed: {e}")
            traceback.print_exc()
            return actions

        # Parse outputs → assign actions
        action_names = {0: "stop", 1: "forward", 2: "left", 3: "right"}
        for agent_i, text in zip(valid_agents, output_texts):
            parsed = self._parse_actions(text)
            if not parsed:
                parsed = [0]
            actions[agent_i]              = parsed[0]
            self.pending_actions[agent_i] = parsed[1:]
            parsed_str = " ".join(action_names.get(a, "?") for a in parsed)
            print(f"  [Infer] agent {agent_i} step {step_idx} | raw: \"{text[:60]}\" → [{parsed_str}]")

        return actions

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _process_grid(vis_embed: torch.Tensor, grid_size: int) -> torch.Tensor:
        """
        Spatial average-pool patch tokens to grid_size × grid_size.
        Input:  (T, num_patches, C)   where num_patches = H*H (square)
        Output: (T, grid_size^2, C)
        Replicates the local process_grid() inside UniNaVid token_generation().
        """
        T, P, C = vis_embed.shape
        H = int(P ** 0.5)
        stride = H // grid_size
        x = vis_embed.reshape(T, H, H, C).permute(0, 3, 1, 2)  # (T, C, H, H)
        x = F.avg_pool2d(x, kernel_size=stride, stride=stride, padding=0)
        return x.permute(0, 2, 3, 1).flatten(1, 2)              # (T, grid_size^2, C)

    def _get_nav_size(self) -> int:
        """Tokens-per-frame stored in history feat_cache."""
        return {"grid:2": 4, "grid:4": 16, "mean": 1}.get(
            self.model.config.compress_type, 4
        )

    def _get_grid_size(self) -> int:
        ct = self.model.config.compress_type
        return int(ct.split("grid:")[-1]) if "grid:" in ct else 2

    def _preprocess_frames(self, rgb_list: list) -> torch.Tensor | None:
        """Preprocess uint8 numpy frames → (T, C, H, W) cuda tensor."""
        if not rgb_list:
            return None
        try:
            batch = np.stack(rgb_list)
        except ValueError:
            return None
        return self.image_processor.preprocess(
            batch, return_tensors="pt"
        )["pixel_values"].to(dtype=self.model.dtype, device=self._device)

    def _build_input_embeds(
        self,
        agent_idx:      int,
        compressed_hist: torch.Tensor,    # (L_hist, Dllm)
        lengths_list:   list[int],
        vis_cur_proj:   torch.Tensor,     # (1, 64, Dllm)
        model_inner,
    ) -> torch.Tensor:
        """
        Manually replicate prepare_inputs_labels_for_multimodal's navigation
        path to produce input_embeds from pre-computed visual features.

        Token layout built by _build_input_ids:
          [text prefix] [VID_START] [IMG_SEP] [-200] [VID_END] [IMG_START] [IMG_END] [NAV_TOK] [suffix]
                                               ^image_token_start (IMAGE_TOKEN_INDEX)

        Resulting input_embeds:
          embed(prefix + VID_START + IMG_SEP)
          + hist_chunk_0 [+ sep + hist_chunk_1 + sep + ...]    ← lengths_list controls chunk sizes
          + embed(VID_END + IMG_START)
          + vis_cur_proj.squeeze(0)                             ← 64 nav tokens, current frame
          + embed(IMG_END + NAV_TOK + text suffix)
        """
        prompt_text = self.PROMPT_TEMPLATE.format(self.instructions[agent_idx])
        cur_ids     = self._build_input_ids(prompt_text)[0]   # (seq_len,)

        img_pos = torch.where(cur_ids == IMAGE_TOKEN_INDEX)[0]
        if img_pos.numel() == 0:
            return model_inner.embed_tokens(cur_ids)

        its     = img_pos[0].item()                           # image_token_start
        sep_emb = model_inner.embed_tokens(cur_ids[its - 1: its])  # (1, Dllm)

        parts: list[torch.Tensor] = []
        parts.append(model_inner.embed_tokens(cur_ids[:its]))       # prefix

        vid_idx = 0
        for k, n_tok in enumerate(lengths_list):
            parts.append(compressed_hist[vid_idx: vid_idx + n_tok])
            if k < len(lengths_list) - 1:
                parts.append(sep_emb)
            vid_idx += n_tok

        parts.append(model_inner.embed_tokens(cur_ids[its + 1: its + 3]))  # VID_END, IMG_START
        parts.append(vis_cur_proj.squeeze(0))                               # 64 nav tokens
        parts.append(model_inner.embed_tokens(cur_ids[its + 3:]))           # IMG_END, NAV_TOK, suffix

        return torch.cat(parts, dim=0)   # (L_i, Dllm)

    def _batch_generate(self, input_embeds_list: list[torch.Tensor]) -> list[str]:
        """
        Left-pad all input_embeds to the same length, then call model.generate()
        once for the entire batch.

        LlavaLlamaAttForCausalLM.forward always calls prepare_inputs_labels_for_multimodal,
        which returns inputs_embeds=None when images=None, discarding our pre-computed
        embeddings.  We temporarily replace model.forward with _bypass_fwd that:
          - On the FIRST generation step (past_key_values is None, inputs_embeds provided):
            directly calls model.model (LlamaModel) and model.lm_head, skipping image coding.
          - On all subsequent autoregressive steps (past_key_values set, inputs_embeds=None):
            delegates to the original forward, which handles single-token decoding normally.
        """
        N = len(input_embeds_list)
        if N == 0:
            return []

        max_len   = max(e.shape[0] for e in input_embeds_list)
        hidden    = input_embeds_list[0].shape[-1]
        batched   = torch.zeros(N, max_len, hidden, dtype=self.model.dtype, device=self._device)
        attn_mask = torch.zeros(N, max_len, dtype=torch.long, device=self._device)
        for i, emb in enumerate(input_embeds_list):
            L = emb.shape[0]
            batched[i, max_len - L:] = emb
            attn_mask[i, max_len - L:] = 1

        from transformers.modeling_outputs import CausalLMOutputWithPast
        _orig_fwd  = self.model.forward
        _model_ref = self.model

        def _bypass_fwd(input_ids=None, attention_mask=None, past_key_values=None,
                        inputs_embeds=None, labels=None, use_cache=None,
                        output_attentions=None, output_hidden_states=None,
                        images=None, prompts=None, return_dict=None, **kw):
            if inputs_embeds is not None and past_key_values is None:
                # First generation step with pre-computed embeds: bypass image encoding
                rd  = return_dict          if return_dict          is not None else _model_ref.config.use_return_dict
                oa  = output_attentions    if output_attentions    is not None else _model_ref.config.output_attentions
                ohs = output_hidden_states if output_hidden_states is not None else _model_ref.config.output_hidden_states
                torch.cuda.empty_cache()
                outputs = _model_ref.model(
                    input_ids=None,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=oa,
                    output_hidden_states=ohs,
                    return_dict=rd,
                )
                logits = _model_ref.lm_head(outputs[0])
                if not rd:
                    return (None, logits) + outputs[1:]
                return CausalLMOutputWithPast(
                    loss=None, logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
            # Subsequent autoregressive steps (past_key_values is set)
            return _orig_fwd(
                input_ids=input_ids, attention_mask=attention_mask,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                labels=labels, use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                images=images, prompts=prompts, return_dict=return_dict,
            )

        self.model.forward = _bypass_fwd
        try:
            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs_embeds=batched,
                    attention_mask=attn_mask,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=128,
                    use_cache=True,
                )
        finally:
            self.model.forward = _orig_fwd

        texts = []
        for i in range(N):
            raw = self.tokenizer.decode(output_ids[i], skip_special_tokens=True).strip()
            if raw.endswith(self.stop_str):
                raw = raw[: -len(self.stop_str)].strip()
            texts.append(raw)
        return texts

    # ------------------------------------------------------------------
    # Helpers (token building / parsing — unchanged)
    # ------------------------------------------------------------------

    def _build_input_ids(self, prompt_text: str) -> torch.Tensor:
        """
        Build (1, seq_len) cuda LongTensor with UniNaVid special tokens inserted
        around the IMAGE_TOKEN_INDEX placeholder.
        """
        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text.replace("<image>", "")
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).to(device=self._device)

        vid_start = self._special_ids["video_start"]
        img_sep   = self._special_ids["image_sep"]
        vid_end   = self._special_ids["video_end"]
        img_start = self._special_ids["image_start"]
        img_end   = self._special_ids["image_end"]
        nav_tok   = self._special_ids["navigation"]

        new_list = []
        indices = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]
        while indices.numel() > 0:
            idx = indices[0]
            new_list.append(token_prompt[:idx])
            new_list.append(vid_start)
            new_list.append(img_sep)
            new_list.append(token_prompt[idx: idx + 1])
            new_list.append(vid_end)
            new_list.append(img_start)
            new_list.append(img_end)
            new_list.append(nav_tok)
            token_prompt = token_prompt[idx + 1:]
            indices = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)

        return torch.cat(new_list, dim=0).unsqueeze(0)  # (1, seq_len)

    def _get_all_special_token_ids(self) -> dict:
        """Pre-resolve all special token ids."""
        token_strs = {
            "video_start": VIDEO_START_SPECIAL_TOKEN,
            "video_end":   VIDEO_END_SPECIAL_TOKEN,
            "image_start": IMAGE_START_TOKEN,
            "image_end":   IMAGE_END_TOKEN,
            "navigation":  NAVIGATION_SPECIAL_TOKEN,
            "image_sep":   IAMGE_SEPARATOR,
        }
        result = {}
        for key, ts in token_strs.items():
            tid = self.tokenizer.convert_tokens_to_ids(ts)
            if tid == self.tokenizer.unk_token_id:
                ids = self.tokenizer(
                    ts, return_tensors="pt", add_special_tokens=False
                ).input_ids[0]
                if len(ids) > 1 and ids[0] == 29871:
                    ids = ids[1:]
                result[key] = ids.cuda()
            else:
                result[key] = torch.tensor([tid], device=self._device)
        return result

    @staticmethod
    def _parse_actions(text: str) -> list[int]:
        """Parse model output into action id list (≤ 2 actions)."""
        w2a = {"stop": 0, "forward": 1, "left": 2, "right": 3}
        actions = []
        for word in text.split():
            if word in w2a:
                actions.append(w2a[word])
                if len(actions) == 2:
                    break
        return actions if actions else [0]


# ---------------------------------------------------------------------------
# Per-scene parallel process
# ---------------------------------------------------------------------------

def _hab_to_genesis(pos_list: list) -> tuple[float, float, float]:
    """Habitat xyz → Genesis xyz 坐标系转换。"""
    x, y, z = pos_list[0], pos_list[1], pos_list[2]
    return x, -z, y


def _genesis_to_hab(cam_pos_gen: torch.Tensor) -> torch.Tensor:
    """cam_pos_gen: (N, 3) Genesis coords → (N, 3) Habitat coords"""
    x = cam_pos_gen[:, 0]
    y = cam_pos_gen[:, 2] - CAMERA_HEIGHT
    z = -cam_pos_gen[:, 1]
    return torch.stack([x, y, z], dim=1)


def _update_camera(cam, cam_pos: torch.Tensor, cam_yaw: torch.Tensor):
    """统一更新 N 个 agent 的相机朝向。"""
    dir_x = torch.cos(cam_yaw)
    dir_y = torch.sin(cam_yaw)
    lookat = cam_pos.clone()
    lookat[:, 0] += dir_x
    lookat[:, 1] += dir_y
    # z 不变（水平朝向）
    lookat[:, 2] = cam_pos[:, 2]
    cam.set_pose(pos=cam_pos, lookat=lookat)


def process_scene(scene_id: str, scene_episodes: list,
                  metrics_queue, scene_loop_idx: int, total_scenes: int):
    """
    在一个独立子进程里运行一个场景的所有 episode（并行）。

    scene_episodes 中的每个 episode 对应一个 Genesis 环境槽位（n_envs=N），
    所有 agent 同步推进，直到全部停止或达到 MAX_STEPS。
    """
    try:
        SCAN_NAME = os.path.basename(os.path.dirname(scene_id))
        N = len(scene_episodes)

        print(f"\n{'#'*60}")
        print(f"Scene {scene_loop_idx}/{total_scenes}: {SCAN_NAME}  ({N} episodes in parallel)")
        print(f"{'#'*60}")

        # ----------------------------------------------------------------
        # 1. Genesis 初始化
        # ----------------------------------------------------------------
        gs.init(backend=gs.cuda)
        renderer = gs.renderers.BatchRenderer(use_rasterizer=True)
        scene = gs.Scene(
            renderer=renderer,
            show_viewer=False,
            vis_options=gs.options.VisOptions(
                ambient_light=(1.0, 1.0, 1.0),
                plane_reflection=False,
            ),
        )

        # ----------------------------------------------------------------
        # 2. 加载场景 mesh
        # ----------------------------------------------------------------
        obj_dir = f"{SCENE_DATASETS}/{SCAN_NAME}/matterport_mesh"
        try:
            obj_files = [f for f in os.listdir(obj_dir) if f.endswith(".obj")]
            if not obj_files:
                raise FileNotFoundError
            obj_path = os.path.join(obj_dir, obj_files[0])
            scene.add_entity(morph=gs.morphs.Mesh(file=obj_path, fixed=True, collision=False))
        except (FileNotFoundError, IndexError):
            print(f"[ERROR] No .obj found for {SCAN_NAME}")
            metrics_queue.put({})
            return

        # ----------------------------------------------------------------
        # 3. 加载 NavMesh → GPU tensors
        # ----------------------------------------------------------------
        navmesh_path = f"{SCENE_DATASETS}/{SCAN_NAME}/navmesh.npz"
        try:
            nm = np.load(navmesh_path)
        except FileNotFoundError:
            print(f"[ERROR] navmesh.npz not found: {navmesh_path}")
            metrics_queue.put({})
            return

        V_hab = torch.tensor(nm["verts"], device=gs.device, dtype=torch.float32)
        F     = torch.tensor(nm["faces"], device=gs.device, dtype=torch.long)

        # Habitat (x, y, z) → Genesis (x, -z, y)
        V_gen = torch.zeros_like(V_hab)
        V_gen[:, 0] =  V_hab[:, 0]
        V_gen[:, 1] = -V_hab[:, 2]
        V_gen[:, 2] =  V_hab[:, 1]

        tri_v0_3d = V_gen[F[:, 0]];  tri_v1_3d = V_gen[F[:, 1]];  tri_v2_3d = V_gen[F[:, 2]]
        tri_v0_2d = tri_v0_3d[:, :2]; tri_v1_2d = tri_v1_3d[:, :2]; tri_v2_2d = tri_v2_3d[:, :2]

        # ----------------------------------------------------------------
        # 4. 建场景 + 相机
        # ----------------------------------------------------------------
        cam = scene.add_camera(res=CAM_RES, fov=FOV, GUI=False)
        scene.build(n_envs=N, env_spacing=(0.0, 0.0))

        # ----------------------------------------------------------------
        # 5. 初始化所有 agent 的位姿（每个 agent 用各自 episode 的初始位姿）
        # ----------------------------------------------------------------
        cam_pos = torch.zeros(N, 3, device=gs.device, dtype=torch.float32)
        cam_yaw = torch.zeros(N,    device=gs.device, dtype=torch.float32)
        instructions   = []
        goal_positions = []   # list of [x, y, z] in Habitat coords
        gt_paths       = []   # list of reference_path

        for i, ep in enumerate(scene_episodes):
            sp = ep["start_position"]
            gx, gy, gz = _hab_to_genesis(sp)
            cam_pos[i] = torch.tensor([gx, gy, gz + CAMERA_HEIGHT],
                                      dtype=torch.float32, device=gs.device)

            rot_q = torch.tensor([ep["start_rotation"]],
                                 dtype=torch.float32, device=gs.device)  # (1, 4)
            cam_yaw[i] = calculate_initial_yaw_tensor(rot_q).squeeze()

            instructions.append(ep["instruction"]["instruction_text"])
            goal_positions.append(ep["goals"][0]["position"])
            gt_paths.append(ep["reference_path"])

        # ----------------------------------------------------------------
        # 6. 初始化 BatchUniNaVidWrapper
        # ----------------------------------------------------------------
        batch_agent = BatchUniNaVidWrapper(MODEL_PATH, N)
        batch_agent.reset_all(instructions)

        # 打印每个 episode 的 header（对标 genesis_uninavid_multi.py 377-380 行）
        for i, ep in enumerate(scene_episodes):
            print(f"\n{'='*60}")
            print(f"Scene {SCAN_NAME} | Slot {i+1}/{N} | Episode ID: {ep['episode_id']}")
            print(f"Instruction: {ep['instruction']['instruction_text']}")
            print(f"{'='*60}")

        # ----------------------------------------------------------------
        # 7. 初始 NavMesh 落地检测 + 相机姿态
        # ----------------------------------------------------------------
        _, _, current_tri_idx = batch_find_floor(
            cam_pos[:, :2],
            cam_pos[:, 2] - CAMERA_HEIGHT,
            tri_v0_2d, tri_v1_2d, tri_v2_2d,
            tri_v0_3d, tri_v1_3d, tri_v2_3d,
        )
        _update_camera(cam, cam_pos, cam_yaw)

        # ----------------------------------------------------------------
        # 8. 初始化 metrics 跟踪
        # ----------------------------------------------------------------
        active_mask = torch.ones(N, dtype=torch.bool, device=gs.device)
        goal_pos_t  = torch.tensor(goal_positions,
                                   dtype=torch.float32, device=gs.device)  # (N, 3) Hab

        pred_path_lists  = [[] for _ in range(N)]
        distances_lists  = [[] for _ in range(N)]

        # 记录初始位置
        init_hab = _genesis_to_hab(cam_pos)  # (N, 3) Habitat
        for i in range(N):
            pred_path_lists[i].append(init_hab[i].cpu().numpy())
            d = torch.norm(init_hab[i] - goal_pos_t[i]).item()
            distances_lists[i].append(d)

        print(f"\n[{SCAN_NAME}] Starting parallel rollout: {N} episodes, max {MAX_STEPS} steps")

        # ----------------------------------------------------------------
        # 9. 主 step 循环
        # ----------------------------------------------------------------
        for step_idx in range(MAX_STEPS):
            n_active = int(active_mask.sum().item())
            if n_active == 0:
                break

            print(f"\n[{SCAN_NAME}] Step {step_idx:04d} | active: {n_active}/{N}")

            # ---- 9a. 并行渲染所有 N 个第一视角 ----
            rgb_batch, _, _, _ = cam.render(
                rgb=True, depth=False, segmentation=False, force_render=True
            )
            # 亮度校正
            if rgb_batch.dtype == torch.uint8:
                rgb_batch = rgb_batch.float() / 255.0
            rgb_batch = torch.clamp(rgb_batch * LIGHT_SCALE * 255.0, 0, 255).byte()
            rgb_np = rgb_batch.cpu().numpy()  # (N, H, W, 3)

            # ---- 9b. 顺序推理（swap cache），一次 step 得到 N 个 action ----
            active_np = active_mask.cpu().numpy()
            actions = batch_agent.step(rgb_np, active_np, step_idx)
            action_t = torch.tensor(actions, device=gs.device, dtype=torch.long)

            # 打印本 step 所有 active agent 的 action
            action_summary = {
                "fwd":  int((action_t[active_mask] == 1).sum()),
                "left": int((action_t[active_mask] == 2).sum()),
                "right":int((action_t[active_mask] == 3).sum()),
                "stop": int((action_t[active_mask] == 0).sum()),
            }
            print(f"  actions → fwd:{action_summary['fwd']} left:{action_summary['left']} "
                  f"right:{action_summary['right']} stop:{action_summary['stop']}")

            # ---- 9c. 判断哪些 agent 本 step 发出 stop ----
            newly_stopped = (action_t == 0) & active_mask
            active_mask[newly_stopped] = False
            # 同步到 batch_agent（下一 step 跳过这些 agent）
            for i in range(N):
                if newly_stopped[i]:
                    batch_agent.pending_actions[i] = []  # 清空残余缓存
                    ep_id  = scene_episodes[i]["episode_id"]
                    dist   = distances_lists[i][-1] if distances_lists[i] else float("nan")
                    print(f"  [Done] slot {i} ep {ep_id} stopped at step {step_idx} | dist_to_goal: {dist:.2f}m")

            if not active_mask.any():
                break

            # ---- 9d. 向量化物理：转向 ----
            left_m  = (action_t == 2) & active_mask
            right_m = (action_t == 3) & active_mask
            cam_yaw[left_m]  += STEP_TURN
            cam_yaw[right_m] -= STEP_TURN

            # ---- 9e. 向量化物理：前进（仅对 active 且 forward 的 agent）----
            fwd_m = (action_t == 1) & active_mask
            if fwd_m.any():
                fwd_idx = fwd_m.nonzero(as_tuple=True)[0]   # 全局索引子集

                dx = torch.cos(cam_yaw[fwd_idx])
                dy = torch.sin(cam_yaw[fwd_idx])
                desired = torch.stack([
                    cam_pos[fwd_idx, 0] + dx * STEP_MOVE,
                    cam_pos[fwd_idx, 1] + dy * STEP_MOVE,
                ], dim=1)                                    # (|fwd|, 2)
                cur_floor_z = cam_pos[fwd_idx, 2] - CAMERA_HEIGHT

                valid, new_z, new_tri = batch_find_floor(
                    desired, cur_floor_z,
                    tri_v0_2d, tri_v1_2d, tri_v2_2d,
                    tri_v0_3d, tri_v1_3d, tri_v2_3d,
                )

                if ALLOW_SLIDING:
                    invalid = ~valid
                    if invalid.any():
                        inv_local = invalid.nonzero(as_tuple=True)[0]
                        slide_pos = batch_get_sliding_position(
                            cam_pos[fwd_idx[inv_local], :2],
                            desired[inv_local],
                            current_tri_idx[fwd_idx[inv_local]],
                            tri_v0_2d, tri_v1_2d, tri_v2_2d,
                        )
                        sv, sz, st = batch_find_floor(
                            slide_pos, cur_floor_z[inv_local],
                            tri_v0_2d, tri_v1_2d, tri_v2_2d,
                            tri_v0_3d, tri_v1_3d, tri_v2_3d,
                        )
                        if sv.any():
                            can_slide = inv_local[sv]
                            desired[can_slide]  = slide_pos[sv]
                            new_z[can_slide]    = sz[sv]
                            new_tri[can_slide]  = st[sv]
                            valid[can_slide]    = True

                moved = fwd_idx[valid]
                cam_pos[moved, 0] = desired[valid, 0]
                cam_pos[moved, 1] = desired[valid, 1]
                cam_pos[moved, 2] = new_z[valid] + CAMERA_HEIGHT
                current_tri_idx[moved] = new_tri[valid]

            # ---- 9f. 更新相机姿态（所有 agent，含 inactive，保持最后位置）----
            _update_camera(cam, cam_pos, cam_yaw)

            # ---- 9g. 记录 metrics（本 step 仍 active 的 agent）----
            curr_hab = _genesis_to_hab(cam_pos)
            for i in range(N):
                if active_mask[i]:
                    pred_path_lists[i].append(curr_hab[i].cpu().numpy())
                    d = torch.norm(curr_hab[i] - goal_pos_t[i]).item()
                    distances_lists[i].append(d)

        # ----------------------------------------------------------------
        # 10. 计算所有 episode 的 metrics
        # ----------------------------------------------------------------
        metrics_all = {}
        for i, ep in enumerate(scene_episodes):
            pred_path    = np.array(pred_path_lists[i])
            gt_path_arr  = np.array(gt_paths[i], dtype=float)
            dists        = distances_lists[i]

            metric = {}
            metric["steps_taken"]       = len(pred_path) - 1
            metric["distance_to_goal"]  = dists[-1]
            metric["success"]           = 1.0 if dists[-1]      <= 3.0 else 0.0
            metric["oracle_success"]    = 1.0 if min(dists)      <= 3.0 else 0.0

            gt_length   = dists[0]
            diffs       = pred_path[1:] - pred_path[:-1]
            path_len    = float(np.linalg.norm(diffs, axis=1).sum()) if len(diffs) else 0.0
            metric["path_length"] = path_len
            metric["spl"] = (
                float(metric["success"] * gt_length / max(gt_length, path_len))
                if max(gt_length, path_len) > 0 else 0.0
            )

            try:
                dtw_dist = fastdtw(pred_path, gt_path_arr, dist=euclidean)[0]
                metric["ndtw"] = float(np.exp(-dtw_dist / (len(gt_path_arr) * 3.0)))
            except Exception:
                metric["ndtw"] = 0.0
            metric["sdtw"] = float(metric["ndtw"] * metric["success"])

            metrics_all[ep["episode_id"]] = metric
            print(f"  ep {ep['episode_id']:>6} | {metric}")

        metrics_queue.put(metrics_all)
        time.sleep(2)
        os._exit(0)

    except Exception as e:
        import traceback
        print(f"[ERROR] process_scene {scene_id}: {e}")
        traceback.print_exc()
        metrics_queue.put({})
        time.sleep(2)
        os._exit(1)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def run_parallel_evaluation():
    episodes = load_data()
    if not episodes:
        print("No episodes loaded.")
        return

    # 按场景分组
    episodes_by_scene: dict[str, list] = {}
    for ep in episodes:
        sid = ep["scene_id"]
        episodes_by_scene.setdefault(sid, []).append(ep)

    # 截断每场景 episode 数（避免渲染 OOM）
    if MAX_PARALLEL_PER_SCENE is not None:
        episodes_by_scene = {
            sid: eps[:MAX_PARALLEL_PER_SCENE]
            for sid, eps in episodes_by_scene.items()
        }

    print(f"Total scenes: {len(episodes_by_scene)}  |  MAX_PARALLEL_PER_SCENE: {MAX_PARALLEL_PER_SCENE or 'unlimited'}")
    for sid, eps in episodes_by_scene.items():
        print(f"  {os.path.basename(os.path.dirname(sid))}: {len(eps)} episodes")

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    metrics_all    = {}
    scene_loop_idx = 0
    total_scenes   = len(episodes_by_scene)

    for scene_id, scene_episodes in episodes_by_scene.items():
        scene_loop_idx += 1
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=process_scene,
            args=(scene_id, scene_episodes, q, scene_loop_idx, total_scenes),
        )
        p.start()
        scene_metrics = q.get()
        p.join()
        if scene_metrics:
            metrics_all.update(scene_metrics)

    # 写出结果
    out_file = "metrics_batch_parallel.json"
    with open(out_file, "w") as f:
        json.dump(metrics_all, f, indent=4)
    print(f"\nSaved metrics to {out_file}")

    if metrics_all:
        keys = next(iter(metrics_all.values())).keys()
        avg  = {k: float(np.mean([m[k] for m in metrics_all.values()])) for k in keys}
        import pprint
        print("\n=== Average Metrics ===")
        pprint.pprint(avg)


if __name__ == "__main__":
    try:
        run_parallel_evaluation()
    except Exception:
        import traceback
        traceback.print_exc()
