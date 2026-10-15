# GenArk 部署文档

> GenArk — 基于 Genesis 仿真器的 VLN 并行评估框架

---

## 一、环境要求

| 项目 | 要求 |
|------|------|
| OS | Ubuntu 20.04+ (已测试) |
| GPU | NVIDIA GPU (Vulkan + CUDA 支持)，推荐 RTX 3090 / A800 |
| CUDA | 12.2+（驱动须支持 CUDA 12.3+，即 Driver ≥ 550.x） |
| Python | 3.11 |
| Conda | Anaconda / Miniconda |
| 磁盘 | 场景数据 ~50GB，模型权重 ~15GB，GLB 缓存 ~300MB/scene |
| 系统依赖 | `libvulkan1`、`ffmpeg` |

> **注意**: Genesis 0.4.3 依赖 `quadrants==0.4.4`，该版本需要 CUDA 12.3+ 驱动（Driver 535.x 对应 CUDA 12.2，会报 `cuGraphConditionalHandleCreate` 符号缺失错误，需升级驱动到 550.x+）。

---

## 二、项目结构

```
genark/
├── orchestrator.py              # 主调度器：启动推理服务 + 分发渲染 worker
├── model_server.py              # FastAPI 推理服务 (UniNaVid LLM)
├── env_worker_glb.py            # Genesis 渲染 worker (GLB 路径，推荐)
├── env_worker.py                # Genesis 渲染 worker (OBJ 路径，备选)
├── genark_config.py             # YAML 配置加载器
├── genark_web_viewer.py         # Web 可视化 (可选)
├── run.sh                       # 启动脚本示例
├── configs/
│   └── uninavid_r2r.yaml        # 评估配置文件 (主配置)
├── model_zoo/                   # 模型权重 (软链接)
│   ├── uninavid-7b-full-224-video-fps-1-grid-2 -> (见下文)
│   └── eva_vit_g.pth -> (见下文)
├── scene_datasets/              # 场景 mesh 数据 (软链接)
├── uninavid/                    # UniNaVid Python 包 (软链接)
├── glb_cache/                   # OBJ→GLB 转换缓存 (自动生成)
├── output_parallel/             # 评估结果输出目录
├── genesis_videos/              # 视频保存目录 (可选)
└── docs/                        # 文档与图表
```

### 软链接说明

项目中有 **3 个关键软链接**，部署时需要正确创建：

| 软链接 | 目标 | 说明 |
|--------|------|------|
| `uninavid/` | `<NAVID_WS>/Uni-NaVid/uninavid` | UniNaVid Python 包，供 model_server.py import |
| `model_zoo/uninavid-7b-*` | `<NAVID_WS>/NaVid-VLN-CE/model_zoo/uninavid-7b-*` | 模型权重 (~14.4 GB, 2 shards) |
| `model_zoo/eva_vit_g.pth` | `<NAVID_WS>/NaVid-VLN-CE/model_zoo/eva_vit_g.pth` | EVA ViT-G 视觉编码器 (~2 GB) |
| `scene_datasets/` | 场景 mesh 数据根目录 | Matterport3D OBJ/NavMesh 文件 |

---

## 三、从零部署步骤

### Step 0: 安装前置系统依赖

```bash
# Miniconda（若未安装）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Vulkan（Genesis 渲染必需）
sudo apt-get install -y libvulkan1 vulkan-tools

# ffmpeg（视频录制必需）
sudo apt-get install -y ffmpeg
# 或无 sudo 时：conda install -n genesis ffmpeg -c conda-forge -y
```

### Step 1: 创建 Conda 环境

```bash
# 首次使用需接受 Anaconda 条款
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n genesis python=3.11 -y
conda activate genesis
```

### Step 2: 安装 PyTorch (CUDA 12.x)

根据实际驱动的 CUDA 版本选择对应 wheel：

```bash
# CUDA 12.2（Driver 535.x）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4+（Driver 550.x+，推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

查看当前 CUDA 版本：`nvidia-smi`

> 已验证版本: PyTorch 2.10.0 + CUDA 12.8（A800，Driver 550+）

### Step 3: 安装 Genesis 仿真器

```bash
cd /path/to/workspace
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
git checkout v0.4.3   # 必须锁定此版本，最新版需要更高 CUDA 驱动
pip install -e .
```

验证（同时验证渲染可用）：
```bash
python -c "import genesis; print(genesis.__version__)"
# 期望输出: 0.4.3

python -c "
import genesis as gs
gs.init(backend=gs.cuda)
scene = gs.Scene()
scene.add_camera(res=(64,64), fov=60, GUI=False)
scene.build()
print('渲染正常')
gs.destroy()
"
```

> **注意**: 不要使用 `git clone` 后直接 `pip install -e .`，必须先 `git checkout v0.4.3`。
> 最新版 Genesis 依赖 `quadrants>=0.6.x`，需要 CUDA 12.4+ 驱动，低版本驱动会报
> `undefined symbol: cuGraphConditionalHandleCreate`。

### Step 4: 克隆 Uni-NaVid

```bash
cd /path/to/workspace
git clone https://github.com/jzhzhang/Uni-NaVid.git
```

> **注意**: 不需要 `pip install -e .`。GenArk 通过软链接直接引用 `uninavid/` 包，
> 以及 `model_server.py` 中的 `sys.path.append` 来解析导入。
> Uni-NaVid 的 `pyproject.toml` 声明了 `torch==2.0.1` 等旧版本依赖，
> 直接安装会引起版本冲突。

### Step 5: 克隆 NaVid-VLN-CE (获取模型权重)

```bash
cd /path/to/workspace
git clone https://github.com/jzhzhang/NaVid-VLN-CE.git
```

#### 下载模型权重

将 UniNaVid-7B 模型权重放入 `NaVid-VLN-CE/model_zoo/` 下：

```
NaVid-VLN-CE/model_zoo/
├── uninavid-7b-full-224-video-fps-1-grid-2/
│   ├── config.json
│   ├── generation_config.json
│   ├── pytorch_model-00001-of-00002.bin    (~9.3 GB)
│   ├── pytorch_model-00002-of-00002.bin    (~5.1 GB)
│   ├── pytorch_model.bin.index.json
│   ├── tokenizer.model
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
└── eva_vit_g.pth                           (~2.0 GB)
```

使用 `hf`（新版 HuggingFace CLI）下载：

```bash
# 登录（只需一次）
hf auth login

# 国内加速（可选）
export HF_ENDPOINT=https://hf-mirror.com

# 下载主模型
hf download Jzzhang/Uni-NaVid \
  --include "uninavid-7b-full-224-video-fps-1-grid-2/*" \
  --local-dir NaVid-VLN-CE/model_zoo/

# 下载 EVA ViT-G（~2GB）
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth \
  -O NaVid-VLN-CE/model_zoo/eva_vit_g.pth
```

> **注意**: `huggingface-cli` 已弃用，新版统一使用 `hf` 命令。

### Step 6: 克隆 GenArk

```bash
cd /path/to/workspace
git clone https://github.com/lukAI-c/genark.git
cd genark
```

### Step 7: 安装 Python 依赖

```bash
pip install fastapi uvicorn pydantic requests python-multipart
pip install transformers==4.31.0 sentencepiece tokenizers
pip install peft accelerate timm
pip install trimesh pygltflib
pip install opencv-python scipy tqdm
pip install fastdtw wandb av
```

> - `transformers` 须锁定 `4.31.0`，该版本与 UniNaVid 的 LLaMA 架构兼容。
> - `python-multipart` 是 FastAPI 的必要依赖，缺少时 model_server 启动报 `RuntimeError: Form data requires python-multipart`。

### Step 8: 创建软链接

假设目录结构如下：
```
/path/to/workspace/
├── Genesis/                 # Step 3
├── Uni-NaVid/               # Step 4
├── NaVid-VLN-CE/            # Step 5
└── genark/                  # Step 6
```

```bash
cd /path/to/workspace/genark

# 1) UniNaVid Python 包
ln -sf /path/to/workspace/Uni-NaVid/uninavid ./uninavid

# 2) 模型权重
mkdir -p model_zoo
ln -sf /path/to/workspace/NaVid-VLN-CE/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2 ./model_zoo/
ln -sf /path/to/workspace/NaVid-VLN-CE/model_zoo/eva_vit_g.pth ./model_zoo/

# 3) 场景数据 (Matterport3D meshes)
ln -sf /path/to/scene_data ./scene_datasets
```

### Step 9: 准备场景数据

`scene_datasets/` 目录下每个场景需包含：

```
scene_datasets/
├── <SCAN_NAME>/
│   ├── matterport_mesh/
│   │   └── *.obj + *.mtl + textures/
│   └── navmesh.npz
├── <SCAN_NAME_2>/
│   └── ...
```

> - `navmesh.npz` 包含 NavMesh 三角面顶点数据（keys: `verts`, `faces`），用于碰撞检测与 sliding
> - 首次运行时，`env_worker_glb.py` 会自动将 OBJ 转换为 GLB 并缓存到 `glb_cache/`

#### 如果只有 Habitat 格式数据（.navmesh + .glb）

若现有场景数据为 Habitat/Matterport3D 标准格式（含 `.navmesh` 二进制文件，无 OBJ），需将 navmesh 转换为 `.npz` 格式：

```bash
# 创建独立转换环境（一次性操作）
conda create -n hab_convert python=3.9 -y
conda activate hab_convert
conda install habitat-sim -c conda-forge -c aihabitat -y
pip install numpy==1.26.4 attrs scipy tqdm numpy-quaternion

# 运行转换脚本（genark 根目录下已有 convert_navmesh.py）
python convert_navmesh.py
```

`convert_navmesh.py` 会将转换结果保存到 `genark/scene_datasets/`，不污染原始数据。

对于 GLB 文件，`env_worker_glb.py` 会优先使用 `glb_cache/` 中已有的缓存；若没有缓存，再从 `scene_datasets/<scan>/matterport_mesh/` 读取 OBJ 进行转换。如果源数据只有 GLB（无 OBJ），需在 `_get_scene_mesh` 函数中添加直接读取源 GLB 的 fallback 逻辑。

### Step 10: 准备数据集

下载 R2R-CE val_unseen 数据集：

```
数据路径示例:
/path/to/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz
```

修改 `configs/uninavid_r2r.yaml` 中的 `EPISODES_FILE` 指向实际路径。

---

## 四、修改 model_server.py 中的硬编码路径

`model_server.py` 第 25 行有硬编码的 `sys.path.append`，需修改为你的实际路径：

```python
# model_server.py line 25
sys.path.append("/path/to/workspace/Uni-NaVid")
```

---

## 五、配置文件说明

编辑 `configs/uninavid_r2r.yaml`：

```yaml
# ---- 数据与模型 ----
EPISODES_FILE:  /path/to/val_unseen.json.gz    # 数据集路径
SCENE_DATASETS: /path/to/genark/scene_datasets  # 场景 mesh 根目录
MODEL_PATH:     model_zoo/uninavid-7b-full-224-video-fps-1-grid-2

# ---- 仿真参数 ----
SIMULATOR:
  CAMERA_HEIGHT:    1.25
  FORWARD_STEP_SIZE: 0.25
  TURN_ANGLE:       30.0
  ALLOW_SLIDING:    true
  AGENT_RADIUS:     0.18
  RGB_SENSOR:
    WIDTH:  640
    HEIGHT: 480
    HFOV:   105
  LIGHT_SCALE: 4.0

TASK:
  MAX_STEPS:       500
  SUCCESS_DISTANCE: 3.0

# ---- 推理参数 ----
GENERATE:
  DO_SAMPLE:       true
  TEMPERATURE:     0.5
  MAX_NEW_TOKENS:  1024

# ---- 并行配置 (根据实际GPU调整) ----
PARALLEL:
  MODEL_GPU:      0              # 推理卡 GPU ID
  WORKER_GPUS:    [1, 2, 3]      # 渲染卡 GPU IDs
  PORT:           8100           # 推理服务端口
  MAX_PARALLEL:   300            # 每 scene 最大并行 episode 数
  MAX_CONCURRENT: null           # 最大并发 worker 数 (null=自动取WORKER_GPUS数)
  CHUNK_SIZE:     8              # 每次 LLM forward 的最大 batch (防 OOM)
  SAVE_VIDEO:     false          # 是否保存 episode 视频
  VIDEO_OUTPUT_DIR: genesis_videos
  VIDEO_FPS:      5
```

### GPU 分配建议

| GPU 配置 | MODEL_GPU | WORKER_GPUS | CHUNK_SIZE |
|---------|-----------|-------------|------------|
| 4× RTX 3090 (24GB) | 0 | [1,2,3] | 8 |
| 8× RTX 3090 (24GB) | 0 | [1,2,3,4,5,6,7] | 8 |
| 4× A800 (80GB) | 0 | [1,2,3] | 1187 |
| 8× A800 (80GB) | 0 | [1,2,3,4,5,6,7] | 1187 |

---

## 六、运行评估

### 方式一: 直接运行

```bash
conda activate genesis
cd /path/to/genark

# 推荐: 设置显存优化 + 关闭代理
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export NO_PROXY=127.0.0.1,localhost

python orchestrator.py --use_glb
```

### 方式二: 使用 run.sh

修改 `run.sh` 中的 GPU 参数后执行：

```bash
bash run.sh
```

### 运行流程

```
orchestrator.py 启动
    │
    ├─ 1) 加载 episode 数据，按 scene 分组
    ├─ 2) 启动 model_server.py (推理卡)
    ├─ 3) 等待推理服务 /health 返回 200
    ├─ 4) 分发 env_worker_glb.py (渲染卡)
    │      ├─ Worker 0 → Scene A (GPU 1)
    │      ├─ Worker 1 → Scene B (GPU 2)
    │      └─ Worker 2 → Scene C (GPU 3)
    ├─ 5) 监控 worker 进度，scene 完成后补位下一个 scene
    └─ 6) 所有 scene 完成 → 汇总 metrics_final.json
```

### 输出结构

```
output_parallel/
└── 20260410_172845__<model>__<dataset>__P300__C3/
    ├── metrics_final.json     # 全部 episode 指标
    ├── avg_metrics.json       # 平均指标汇总
    ├── run_config.json        # 运行参数 + 总耗时
    └── logs/
        ├── model_server.log   # 推理服务日志
        ├── worker_0.log       # 渲染 worker 日志
        ├── worker_0_progress.json
        └── ...
```

---

## 七、常见问题

### Q1: 推理服务启动后卡在 "Waiting for model server"

**原因**: 代理（HTTP_PROXY）拦截了本地 127.0.0.1 请求。

**解决**:
```bash
export NO_PROXY=127.0.0.1,localhost
# 或启动时:
NO_PROXY=127.0.0.1,localhost python orchestrator.py --use_glb
```

### Q2: model_server OOM

**原因**: `CHUNK_SIZE` 过大，超过推理卡显存。

**解决**: 在 `configs/uninavid_r2r.yaml` 中减小 `CHUNK_SIZE`（RTX 3090 建议 8，A800 可到 1000+）。

### Q3: 部分场景渲染全黑

**原因**: OBJ 纹理格式与 Genesis Vulkan 管线不兼容。

**解决**: 使用 `--use_glb` 参数启动（推荐），会自动将 OBJ 转为 GLB。首次转换会在 `glb_cache/` 目录生成缓存。

### Q4: Agent 卡死不动（持续输出同一动作）

**原因**: NavMesh sliding 逻辑在边界处失效，agent 嵌入 mesh。

**解决**: 已在 `env_worker_glb.py` 中加入碰撞回退机制（`agent_radius` 内缩），无需额外配置。

### Q5: GLB 缓存磁盘空间

每个 Matterport3D 场景转 GLB 约 20~55 MB，11 个 val_unseen 场景共 ~300 MB。缓存存储在 `glb_cache/`，可手动清理后重新生成。

### Q6: `RuntimeError: Form data requires python-multipart`

**原因**: FastAPI 依赖 `python-multipart` 但未安装。

**解决**: `pip install python-multipart`

### Q7: `undefined symbol: cuGraphConditionalHandleCreate`（Genesis Worker 崩溃）

**原因**: Genesis 依赖的 `quadrants` 库需要 CUDA 12.3+ 驱动，Driver 535.x（CUDA 12.2）不支持。

**解决**: 升级 NVIDIA 驱动至 550.x+（支持 CUDA 12.4+），或锁定 Genesis 到 `v0.4.3`（依赖 `quadrants==0.4.4`）并使用 CUDA 12.4+ 驱动。

### Q8: `ValueError: Not find vision tower: ./model_zoo/eva_vit_g.pth`

**原因**: `eva_vit_g.pth` 文件不存在，模型 config.json 中路径为相对路径 `./model_zoo/eva_vit_g.pth`，从 genark 根目录运行时须确保该文件存在。

**解决**: 下载 `eva_vit_g.pth` 并放入 `NaVid-VLN-CE/model_zoo/`，确认软链接 `genark/model_zoo/eva_vit_g.pth` 指向正确。

### Q9: `[ERROR] navmesh.npz not found`

**原因**: 场景目录下没有 `navmesh.npz`，或 `env_worker_glb.py` 中 `SCENE_DATASETS` 路径未从配置文件读取（旧版有硬编码路径）。

**解决**:
1. 检查 `env_worker_glb.py` 第 108 行，确认 `SCENE_DATASETS` 从配置读取而非硬编码
2. 若源数据为 Habitat `.navmesh` 格式，使用 `convert_navmesh.py` 脚本转换（见 Step 9）

### Q10: 运行结束但 `genesis_videos/` 目录为空（无 MP4）

**原因**: `ffmpeg` 未安装，视频帧编译失败，`finally` 块清理了临时帧目录。

**解决**: `sudo apt-get install -y ffmpeg`（或 `conda install -n genesis ffmpeg -c conda-forge -y`），重新运行。

---

## 八、依赖版本汇总

| 包名 | 版本 | 备注 |
|------|------|------|
| genesis-world | 0.4.3 | 从源码 editable 安装 |
| torch | 2.10.0+cu128 | CUDA 12.8 |
| transformers | 4.31.0 | 须锁定此版本 |
| fastapi | 0.135.2 | 推理服务 HTTP 框架 |
| uvicorn | 0.42.0 | ASGI 服务器 |
| peft | 0.18.1 | LoRA 支持 |
| accelerate | 1.13.0 | HuggingFace 加速 |
| timm | 1.0.25 | 视觉模型库 |
| trimesh | 4.11.4 | OBJ→GLB 转换 |
| pygltflib | — | GLB 读写 |
| opencv-python | 4.13.0 | 图像处理 |
| scipy | 1.17.1 | 科学计算 |
| fastdtw | 0.3.4 | DTW 指标计算 |
| sentencepiece | 0.2.1 | Tokenizer |
| tqdm | 4.67.3 | 进度条 |
| wandb | 0.25.1 | 实验追踪 (可选) |
