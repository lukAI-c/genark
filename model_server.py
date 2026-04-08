#!/usr/bin/env python3
"""
model_server.py — UniNaVid batch inference service.

One process, one GPU, serves multiple Genesis env-workers over HTTP.
Image data is sent as raw binary (no base64) via multipart upload.

Usage:
    CUDA_VISIBLE_DEVICES=5 python model_server.py --port 8100 --model_path model_zoo/uninavid-7b-full-224-video-fps-1-grid-2

Endpoints:
    GET  /health                       → server status + session count
    POST /create_session               → create per-scene session (JSON body)
    POST /step                         → one simulation step (multipart: rgb bytes + metadata)
    DELETE /session/{session_id}       → cleanup
"""

import sys
import os
import uuid
import threading
import argparse

sys.path.append("/data1/home/clk/project/Genesis")
sys.path.append("/data1/home/clk/navid_ws/Uni-NaVid")

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn

from transformers.modeling_outputs import CausalLMOutputWithPast

from uninavid.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from uninavid.model.builder import load_pretrained_model
from uninavid.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    VIDEO_START_SPECIAL_TOKEN, VIDEO_END_SPECIAL_TOKEN,
    IMAGE_START_TOKEN, IMAGE_END_TOKEN,
    NAVIGATION_SPECIAL_TOKEN, IAMGE_SEPARATOR,
)
from uninavid.conversation import conv_templates, SeparatorStyle


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Loads UniNaVid once.  Session state (feat_cache, rgb_lists, …) is kept
    externally as a plain dict so multiple scenes can share the same weights.

    Thread-safety: callers must hold _global_lock before calling step().
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

    def __init__(self, model_path: str, chunk_size: int = 16,
                 do_sample: bool = True, temperature: float = 0.2,
                 max_new_tokens: int = 128):
        self.chunk_size = chunk_size
        self.do_sample = do_sample
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.conv_mode = "vicuna_v1"
        print(f"[Engine] Loading UniNaVid from {model_path} …")
        model_name = get_model_name_from_path(model_path)
        # Force all layers onto a single GPU to avoid cross-device tensor issues
        # when _bypass_fwd short-circuits HuggingFace's AlignDevicesHook.
        (self.tokenizer, self.model,
         self.image_processor, self.context_len) = load_pretrained_model(
            model_path, None, model_name, device_map={"": 0}
        )
        self.model.eval()
        self.model.config.run_type = "eval"
        self._device = next(self.model.parameters()).device
        print(f"[Engine] Model loaded on {self._device}")

        _conv = conv_templates[self.conv_mode].copy()
        self.stop_str = (
            _conv.sep if _conv.sep_style != SeparatorStyle.TWO else _conv.sep2
        )
        self._special_ids = self._get_all_special_token_ids()

    # ------------------------------------------------------------------
    # Session state factory
    # ------------------------------------------------------------------

    def new_state(self, n_agents: int, instructions: list[str]) -> dict:
        """Create the per-session mutable state dict."""
        assert len(instructions) == n_agents
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        return {
            "n_agents":          n_agents,
            "instructions":      list(instructions),
            "rgb_lists":         [[] for _ in range(n_agents)],
            "pending_actions":   [[] for _ in range(n_agents)],
            "feat_caches":       [None] * n_agents,
            "long_feat_caches":  [None] * n_agents,
            "weights":           [1]    * n_agents,
        }

    # ------------------------------------------------------------------
    # Public API  (call under _global_lock)
    # ------------------------------------------------------------------

    def step(self, state: dict,
             rgb_batch_np: np.ndarray,
             active: np.ndarray,
             step_idx: int) -> list[int]:
        """
        Run one navigation step for all agents in this session.

        Phase 1 – consume buffered actions / collect agents needing inference.
        Phase 2 – batch vision encoding (one vision-tower forward).
        Phase 3 – per-agent cache update + build input_embeds.
        Phase 4 – one batched LLM generate call.
        """
        n = state["n_agents"]
        actions = [0] * n

        # ---- Phase 1: consume pending actions ----
        need_infer: list[int] = []
        for i in range(n):
            if not active[i]:
                continue
            state["rgb_lists"][i].append(rgb_batch_np[i].copy())
            if state["pending_actions"][i]:
                actions[i] = state["pending_actions"][i].pop(0)
            else:
                need_infer.append(i)

        if not need_infer:
            return actions

        # ---- Phase 2: preprocess + batch vision encoding ----
        video_tensors: list[torch.Tensor] = []
        frame_counts:  list[int]          = []
        valid_agents:  list[int]          = []
        for i in need_infer:
            vt = self._preprocess_frames(state["rgb_lists"][i])
            if vt is not None:
                video_tensors.append(vt)
                frame_counts.append(vt.shape[0])
                valid_agents.append(i)
            else:
                actions[i] = 0

        if not video_tensors:
            return actions

        model_inner = self.model.get_model()
        nav_size    = self._get_nav_size()
        grid_size   = self._get_grid_size()

        try:
            with torch.inference_mode():
                all_frames = torch.cat(video_tensors, dim=0)         # (ΣTi, C, H, W)
                all_raw    = model_inner.get_vision_tower()(all_frames)  # (ΣTi, P, Dv)
                if (self.model.config.mm_vision_select_feature == "patch"
                        and all_raw.shape[1] % 2 == 1):
                    all_raw = all_raw[:, 1:]                          # remove CLS token

                # ---- Phase 3: per-agent cache update + input_embeds ----
                input_embeds_list: list[torch.Tensor] = []
                offset = 0
                for agent_i, Ti in zip(valid_agents, frame_counts):
                    agent_raw = all_raw[offset: offset + Ti]

                    vis_hist = self._process_grid(agent_raw, grid_size)   # (Ti, nav_size, Dv)
                    vis_cur  = self._process_grid(agent_raw[-1:], 8)      # (1, 64, Dv)

                    vis_hist_proj = model_inner.mm_projector(vis_hist)
                    vis_cur_proj  = model_inner.mm_projector(vis_cur)

                    # Swap in this agent's cache
                    model_inner.feat_cache      = state["feat_caches"][agent_i]
                    model_inner.long_feat_cache = state["long_feat_caches"][agent_i]
                    model_inner.weight          = state["weights"][agent_i]
                    model_inner.new_frames      = Ti

                    old = model_inner.feat_cache
                    model_inner.feat_cache = (
                        torch.cat([old, vis_hist_proj], dim=0)
                        if old is not None else vis_hist_proj
                    )

                    compressed, lengths = self.model.online_process_tensor(nav_size)

                    embeds_i = self._build_input_embeds(
                        state["instructions"][agent_i],
                        compressed, lengths, vis_cur_proj, model_inner,
                    )
                    input_embeds_list.append(embeds_i)

                    # Swap out
                    state["feat_caches"][agent_i]      = model_inner.feat_cache
                    state["long_feat_caches"][agent_i] = model_inner.long_feat_cache
                    state["weights"][agent_i]          = model_inner.weight
                    model_inner.feat_cache             = None
                    model_inner.long_feat_cache        = None
                    model_inner.weight                 = 1
                    model_inner.new_frames             = 0

                    state["rgb_lists"][agent_i] = []
                    offset += Ti

                # ---- Phase 4: one batched LLM generate call ----
                output_texts = self._batch_generate(input_embeds_list)

        except Exception as exc:
            import traceback
            print(f"[Engine] step {step_idx} batch inference failed: {exc}")
            traceback.print_exc()
            return actions

        action_names = {0: "stop", 1: "forward", 2: "left", 3: "right"}
        log_parts = []
        for agent_i, text in zip(valid_agents, output_texts):
            parsed = self._parse_actions(text)
            if not parsed:
                parsed = [0]
            actions[agent_i]                   = parsed[0]
            state["pending_actions"][agent_i]  = parsed[1:]
            parsed_str = " ".join(action_names.get(a, "?") for a in parsed)
            # print(f"  [Infer] agent {agent_i} step {step_idx} | raw: \"{text[:60]}\" → [{parsed_str}]",
            #       flush=True)
            # log_parts.append(f"A{agent_i}:[{parsed_str}]")
        # print(f"[Server] Model Infer Step {step_idx} -> " + " | ".join(log_parts), flush=True)

        return actions

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _process_grid(vis_embed: torch.Tensor, grid_size: int) -> torch.Tensor:
        T, P, C = vis_embed.shape
        H = int(P ** 0.5)
        stride = H // grid_size
        x = vis_embed.reshape(T, H, H, C).permute(0, 3, 1, 2)
        x = F.avg_pool2d(x, kernel_size=stride, stride=stride, padding=0)
        return x.permute(0, 2, 3, 1).flatten(1, 2)

    def _get_nav_size(self) -> int:
        return {"grid:2": 4, "grid:4": 16, "mean": 1}.get(
            self.model.config.compress_type, 4
        )

    def _get_grid_size(self) -> int:
        ct = self.model.config.compress_type
        return int(ct.split("grid:")[-1]) if "grid:" in ct else 2

    def _preprocess_frames(self, rgb_list: list) -> torch.Tensor | None:
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
        instruction:     str,
        compressed_hist: torch.Tensor,
        lengths_list:    list[int],
        vis_cur_proj:    torch.Tensor,
        model_inner,
    ) -> torch.Tensor:
        prompt_text = self.PROMPT_TEMPLATE.format(instruction)
        cur_ids     = self._build_input_ids(prompt_text)[0]

        img_pos = torch.where(cur_ids == IMAGE_TOKEN_INDEX)[0]
        if img_pos.numel() == 0:
            return model_inner.embed_tokens(cur_ids)

        its     = img_pos[0].item()
        sep_emb = model_inner.embed_tokens(cur_ids[its - 1: its])

        parts: list[torch.Tensor] = []
        parts.append(model_inner.embed_tokens(cur_ids[:its]))

        vid_idx = 0
        for k, n_tok in enumerate(lengths_list):
            parts.append(compressed_hist[vid_idx: vid_idx + n_tok])
            if k < len(lengths_list) - 1:
                parts.append(sep_emb)
            vid_idx += n_tok

        parts.append(model_inner.embed_tokens(cur_ids[its + 1: its + 3]))
        parts.append(vis_cur_proj.squeeze(0))
        parts.append(model_inner.embed_tokens(cur_ids[its + 3:]))

        return torch.cat(parts, dim=0)

    def _batch_generate(self, input_embeds_list: list[torch.Tensor]) -> list[str]:
        N = len(input_embeds_list)
        if N == 0:
            return []

        CHUNK_SIZE = self.chunk_size
        all_texts = []

        _orig_fwd  = self.model.forward
        _model_ref = self.model

        def _bypass_fwd(input_ids=None, attention_mask=None, past_key_values=None,
                        inputs_embeds=None, labels=None, use_cache=None,
                        output_attentions=None, output_hidden_states=None,
                        images=None, prompts=None, return_dict=None, **kw):
            
            # 无论 Prefill(有 inputs_embeds) 还是 Decode(有 input_ids 和 past_key_values)
            # 都全部直通底层纯语言模型，彻底绕过会导致 'NoneType' 报错的外层多模态 _orig_fwd
            rd  = return_dict          if return_dict          is not None else _model_ref.config.use_return_dict
            oa  = output_attentions    if output_attentions    is not None else _model_ref.config.output_attentions
            ohs = output_hidden_states if output_hidden_states is not None else _model_ref.config.output_hidden_states
            
            outputs = _model_ref.model(
                input_ids=input_ids,
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

        self.model.forward = _bypass_fwd

        try:
            for chunk_start in range(0, N, CHUNK_SIZE):
                chunk_embeds = input_embeds_list[chunk_start : chunk_start + CHUNK_SIZE]
                chunk_N      = len(chunk_embeds)
                
                max_len   = max(e.shape[0] for e in chunk_embeds)
                hidden    = chunk_embeds[0].shape[-1]
                batched   = torch.zeros(chunk_N, max_len, hidden,
                                        dtype=self.model.dtype, device=self._device)
                attn_mask = torch.zeros(chunk_N, max_len, dtype=torch.long, device=self._device)
                
                for i, emb in enumerate(chunk_embeds):
                    L = emb.shape[0]
                    batched[i, max_len - L:]   = emb
                    attn_mask[i, max_len - L:] = 1

                # Build stopping criteria (stop at </s>)
                # Use a dummy input_ids for KeywordsStoppingCriteria start_len
                dummy_ids = torch.zeros(chunk_N, max_len, dtype=torch.long,
                                        device=self._device)
                stopping = KeywordsStoppingCriteria(
                    [self.stop_str], self.tokenizer, dummy_ids
                )

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        inputs_embeds=batched,
                        attention_mask=attn_mask,
                        do_sample=self.do_sample,
                        temperature=self.temperature,
                        max_new_tokens=self.max_new_tokens,
                        use_cache=True,
                        stopping_criteria=[stopping],
                    )

                for i in range(chunk_N):
                    raw = self.tokenizer.decode(output_ids[i], skip_special_tokens=True).strip()
                    if raw.endswith(self.stop_str):
                        raw = raw[: -len(self.stop_str)].strip()
                    all_texts.append(raw)
                    
                torch.cuda.empty_cache()  # 每一个小批次推理完成后清理缓存释放显存
                
        finally:
            # 无论成功失败，确保恢复模型原始的 forward
            self.model.forward = _orig_fwd

        return all_texts

    def _build_input_ids(self, prompt_text: str) -> torch.Tensor:
        qs   = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text.replace("<image>", "")
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
        indices  = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]
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
            indices      = torch.where(token_prompt == IMAGE_TOKEN_INDEX)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)

        return torch.cat(new_list, dim=0).unsqueeze(0)

    def _get_all_special_token_ids(self) -> dict:
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
                result[key] = ids.to(self._device)
            else:
                result[key] = torch.tensor([tid], device=self._device)
        return result

    @staticmethod
    def _parse_actions(text: str) -> list[int]:
        w2a = {"stop": 0, "forward": 1, "left": 2, "right": 3}
        actions = []
        for word in text.split():
            if word in w2a:
                actions.append(w2a[word])
                if word == "stop" or len(actions) == 2:
                    break
        return actions if actions else [0]


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app     = FastAPI(title="UniNaVid Inference Server")
_engine: InferenceEngine | None = None
_sessions: dict[str, dict]      = {}
_global_lock                    = threading.Lock()  # serialises all model use


class CreateSessionRequest(BaseModel):
    n_agents:     int
    instructions: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(_sessions)}


@app.post("/create_session")
def create_session(req: CreateSessionRequest):
    if _engine is None:
        raise HTTPException(503, "Engine not initialised")
    sid   = str(uuid.uuid4())
    state = _engine.new_state(req.n_agents, req.instructions)
    with _global_lock:
        _sessions[sid] = state
    print(f"[Server] Session created: {sid}  n_agents={req.n_agents}", flush=True)
    return {"session_id": sid}


@app.post("/step")
async def step_endpoint(
    session_id: str  = Form(),
    step_idx:   int  = Form(),
    shape_str:  str  = Form(),   # e.g. "8 480 640 3"
    active_str: str  = Form(),   # e.g. "1 1 0 1 1 1 1 1"
    rgb:        UploadFile = File(),
):
    if _engine is None:
        raise HTTPException(503, "Engine not initialised")
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found")

    shape    = tuple(int(x) for x in shape_str.split())
    active   = np.array([int(x) for x in active_str.split()], dtype=bool)
    rgb_bytes = await rgb.read()
    rgb_np   = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(shape)

    state = _sessions[session_id]
    with _global_lock:
        actions = _engine.step(state, rgb_np, active, step_idx)

    return {"actions": actions}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    with _global_lock:
        _sessions.pop(session_id, None)
    print(f"[Server] Session deleted: {session_id}", flush=True)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port",       type=int,  default=8100)
    p.add_argument("--host",       type=str,  default="0.0.0.0")
    p.add_argument("--model_path", type=str,
                   default="model_zoo/uninavid-7b-full-224-video-fps-1-grid-2")
    p.add_argument("--chunk_size", type=int,  default=16,
                   help="Max batch size per LLM generate call")
    p.add_argument("--do_sample",      type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--temperature",    type=float, default=0.2)
    p.add_argument("--max_new_tokens", type=int,   default=128)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _engine = InferenceEngine(
        args.model_path,
        chunk_size=args.chunk_size,
        do_sample=args.do_sample,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    # uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)
