# GenArk: Genesis-Habitat Navigation Scripts

在 Genesis 仿真器中运行 Habitat 导航 episode 的脚本集，包含两个功能：
1. **GT 动作并行重放**：用 Ground Truth 动作批量驱动 agent（无需模型）
2. **UniNaVid 模型推理评估**：加载 UniNaVid 模型串行推理，支持多场景自动切换

## 目录结构

```
genark/
├── genesis_gt_path_multi.py       # GT 动作并行重放（无需模型）
├── agent_uninavid_genesis.py      # UniNaVid Agent 封装类
├── genesis_uninavid_multi.py      # UniNaVid 多场景串行评估（import agent_uninavid_genesis）
└── README.md
```

## 前置依赖

### 1. Genesis 仿真器

```bash
pip install genesis-world
```

需要 CUDA 支持，Genesis 使用 GPU 进行批量渲染。

### 2. Python 依赖

```bash
# 两个脚本共需
pip install torch numpy opencv-python

# genesis_uninavid_multi.py 额外需要
pip install fastdtw scipy
```

### 3. 场景数据

每个 Matterport3D 场景需要以下文件结构：

```
<SCENE_DATASETS>/<SCAN_NAME>/
├── matterport_mesh/
│   └── *.obj          # 3D 场景 mesh
└── navmesh.npz        # 导航网格（包含 verts 和 faces）
```

### 4. Episode 数据

JSON 格式，字段说明：

| 字段 | GT 脚本 | UniNaVid 脚本 | 说明 |
|------|:---:|:---:|------|
| `episode_id` | 需要 | 需要 | 唯一 ID |
| `scene_id` | 需要 | 需要 | 场景路径（用于提取 SCAN_NAME） |
| `start_position` / `start_rotation` | 需要 | 需要 | 起始位姿（Habitat 坐标系） |
| `actions` | 需要 | - | GT 动作序列 (0=stop, 1=forward, 2=left, 3=right) |
| `locations` | 需要 | - | GT 路径点序列 |
| `goals[0].position` | - | 需要 | 目标位置 |
| `reference_path` | - | 需要 | GT 路径（用于计算 DTW） |
| `instruction.instruction_text` | - | 需要 | 导航指令文本 |

> 注意：两个脚本的 episode JSON 格式不同。GT 脚本使用扁平结构（`goal_position`, `instruction_text`），UniNaVid 脚本使用嵌套结构（`goals[0].position`, `instruction.instruction_text`）。

### 5. UniNaVid 模型（仅推理脚本需要）

- `uninavid` 包需要在 Python path 中可 import（脚本会自动将上两级目录加入 `sys.path`）
- 模型权重默认路径：`model_zoo/uninavid-7b-full-224-video-fps-1-grid-2/`
- 可选：`model_zoo/eva_vit_g.pth`（Vision Tower 权重，模型可能依赖）

---

## 脚本一：GT 动作并行重放

**文件**：`genesis_gt_path_multi.py`（单文件，无其他依赖）

用 Ground Truth 动作序列驱动 agent 在 Genesis 中批量走完所有 episode。不加载任何模型。适合数据采集、环境验证、渲染效果检查。

### 运行方式

```bash
# 修改脚本顶部路径后运行
python genesis_gt_path_multi.py
```

### 需要修改的配置

```python
EPISODES_FILE = "/path/to/mp3d_<scene_name>.json"
SCENE_DATASETS = "/path/to/scene_datasets"
NUM_PARALLEL_ENVS = 1024   # 根据 GPU 显存调整，不加载模型可以开很大
SAVE_IMAGES = False         # 设为 True 保存渲染图到 output/episode_<id>/
```

### 执行流程

```
加载同一场景的 N 个 episode
→ Genesis 创建 N 个并行环境 (n_envs=N)
→ 每步按 GT 动作批量移动所有 agent + 批量渲染
→ 所有 episode 完成后输出 FPS 统计
```

### 输出

- 终端打印仿真速度（FPS）和各阶段耗时
- 若 `SAVE_IMAGES=True`，图像保存到 `output/episode_<id>/step_XXXX.png`

---

## 脚本二：UniNaVid 模型推理评估

**文件**：`genesis_uninavid_multi.py` + `agent_uninavid_genesis.py`

加载 UniNaVid 模型，对每个 episode 逐步渲染观测、推理动作、执行导航，最终计算评估指标。支持多场景自动切换。

### 运行方式

```bash
# 修改脚本顶部路径后运行
python genesis_uninavid_multi.py
```

### 需要修改的配置

```python
EPISODES_FILE = "/path/to/episodes.json"   # 可包含多个场景的 episode
SCENE_DATASETS = "/path/to/scene_datasets"
MODEL_PATH = "model_zoo/uninavid-7b-full-224-video-fps-1-grid-2"
NUM_EPISODES = 100          # 要评估的 episode 总数
SAVE_DEBUG_IMAGES = False   # 设为 True 保存 agent 每步的输入图像
```

### 执行流程

```
加载所有 episode → 按 scene_id 分组
→ 逐场景循环：
    spawn 子进程（绕开 Genesis 不能重复 init 的限制）
    → 加载该场景 mesh + navmesh
    → 初始化 UniNaVid Agent（加载模型权重）
    → 逐 episode 串行：
        agent.reset()
        → while 循环（最多 500 步）：
            渲染 RGB → agent.act() → 得到动作
            → NavMesh 碰撞检测 + sliding → 移动
        → 计算该 episode 的 metrics
    → 通过 Queue 将 metrics 传回主进程
    → 子进程 os._exit(0) 强制退出（避免 CUDA 清理错误）
→ 汇总所有场景 metrics → 输出 JSON + 终端打印平均值
```

> **为什么用子进程？** Genesis 的 `gs.init()` 在同一进程中不能多次调用。切换场景需要重新初始化，因此每个场景在独立子进程中运行。

### 输出

- `metrics_multi_scene_eval.json`：每个 episode 的详细指标
- 终端打印平均指标汇总

### 评估指标

| 指标 | 含义 |
|------|------|
| `success` | 终点距目标 <= 3m 为成功 |
| `oracle_success` | 路径上任意点距目标 <= 3m |
| `spl` | Success weighted by Path Length |
| `path_length` | Agent 实际行走路径长度 |
| `distance_to_goal` | 终点到目标的欧氏距离 |
| `ndtw` | Normalized Dynamic Time Warping（路径相似度） |
| `sdtw` | Success weighted DTW |
| `steps_taken` | 实际执行步数 |

---

## 参数参考

### 通用参数

| 参数 | GT 脚本默认值 | UniNaVid 脚本默认值 | 说明 |
|------|:---:|:---:|------|
| `CAMERA_HEIGHT` | 0.88 | 0.88 | 相机高度（米） |
| `CAM_RES` | (640, 480) | (640, 480) | 渲染分辨率 (宽, 高) |
| `FOV` | 64 | 64 | 相机视场角（度） |
| `LIGHT_SCALE` | 4.0 | 4.0 | 光照强度缩放 |
| `STEP_MOVE` | 0.25 | 0.25 | 前进步长（米） |
| `STEP_TURN` | 15 | 30 | 转弯角度（度） |
| `ALLOW_SLIDING` | True | True | 碰撞时是否沿墙滑动 |
| `MAX_STEP_HEIGHT` | 0.5 | 0.5 | 最大可跨越高度差（米） |

> 注意：两个脚本的 `STEP_TURN` 默认值不同（GT: 15, UniNaVid: 30）。

## 坐标系

脚本内部自动处理 Habitat 与 Genesis 的坐标系转换：

```
Habitat (x, y, z)  →  Genesis (x, -z, y)
```

- Habitat: x-right, y-up, z-backward
- Genesis: x-right, y-forward, z-up

## FAQ

**显存不够？**
GT 脚本：降低 `NUM_PARALLEL_ENVS` 或 `CAM_RES`。UniNaVid 脚本固定 `n_envs=1`，显存主要被模型占用。

**渲染图像太暗/太亮？**
调整 `LIGHT_SCALE`，默认 4.0 是对 Genesis 渲染亮度的补偿。

**UniNaVid 脚本能并行多个环境吗？**
当前不能。UniNaVid 推理是逐 episode 串行的（`n_envs=1`）。多场景通过子进程串行切换，不是并行。

**子进程为什么用 `os._exit(0)`？**
Genesis/CUDA 在正常 Python 退出流程中可能触发 double-free 错误，`os._exit()` 跳过清理直接退出。
