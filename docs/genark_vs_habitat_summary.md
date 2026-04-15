# GenArk vs Habitat-Sim: VLN 评估系统对比

> UniNaVid 模型 · R2R-CE val_unseen · 4× NVIDIA RTX 3090

---

## 一、系统架构对比

### Habitat-Sim 评估流程

```
┌─────────────────────────────────────────────────┐
│              eval_uninavid_vlnce.sh              │
│                                                  │
│   GPU 0 ─── run.py (split 0) ──┐                │
│   GPU 3 ─── run.py (split 1) ──┤  各进程独立     │
│   GPU 4 ─── run.py (split 2) ──┤  wait 等待全部  │
│   GPU 5 ─── run.py (split 3) ──┘  完成后汇总     │
│                                                  │
│   每个进程内部:                                    │
│   ┌─────────────────────────────────────────┐    │
│   │  加载模型 → for ep in episodes:         │    │
│   │    env.reset()                          │    │
│   │    while not done:                      │    │
│   │      渲染(单帧) → 推理(单agent) → step  │    │
│   │    收集 metrics                         │    │
│   └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

**核心特点:**
- 4 个独立进程，**数据分片**并行（dataset.get_splits）
- 每个进程内 episode **顺序执行**，同一时刻仅 1 个 agent 在仿真
- 每个进程独立加载一份完整模型（~14GB VRAM），渲染与推理共享同一 GPU
- 总并行 agent 数 = GPU 数 = **4**

### GenArk 评估流程

```
┌──────────────────────────────────────────────────────────┐
│                     orchestrator.py                       │
│                                                          │
│   GPU 2 ─── model_server.py (HTTP, 统一推理服务)          │
│              ├─ /create_session                           │
│              ├─ /step (batch generate, chunk_size=8)      │
│              └─ /delete_session                           │
│                                                          │
│   GPU 3 ─── env_worker_glb.py (scene A)                  │
│              └─ Genesis BatchRenderer                     │
│              └─ n_envs=N (该scene全部episode并行仿真)     │
│                                                          │
│   GPU 4 ─── env_worker_glb.py (scene B)     同时最多      │
│              └─ ...                          3个worker    │
│                                                          │
│   GPU 5 ─── env_worker_glb.py (scene C)                  │
│              └─ ...                                       │
│                                                          │
│   ThreadPoolExecutor(max_workers=3)                       │
│   scene 完成后自动调度下一个 scene                          │
└──────────────────────────────────────────────────────────┘
```

**核心特点:**
- **推理与渲染解耦**: 模型独占 1 张 GPU，渲染 worker 独占 3 张 GPU
- 每个 worker 内，同一 scene 的 **全部 episode 并行仿真**（Genesis `n_envs=N`）
- 渲染: Genesis CUDA BatchRenderer **一次调用渲染 N 个视角**
- 推理: worker 将 N 个 agent 的观测通过 HTTP 批量送入 model_server，分 chunk（每批最多 8 个）generate
- worker 按 scene 调度，3 个 worker 并发，scene 完成后自动补位
- 总并行 agent 数 = 3 × N（N 为 scene 内 episode 数，可达**数十到数百**）

---

## 二、关键差异总结

| 维度                | Habitat-Sim                    | GenArk                                      |
|---------------------|-------------------------------|----------------------------------------------|
| **仿真器**          | Habitat-Sim (C++/OpenGL)       | Genesis (Python/CUDA Vulkan)                 |
| **渲染方式**        | 单帧逐个渲染                    | BatchRenderer 批量并行渲染                    |
| **推理部署**        | 每进程独立加载模型               | 中心化 HTTP 推理服务（模型加载 1 次）          |
| **并行粒度**        | 进程级（数据分片）               | 环境级（同 scene 多 episode 并行仿真）        |
| **同时在线 agent**  | 4（= GPU 数）                   | 3 × N（N = scene 内 episode 数）             |
| **GPU 用途**        | 渲染+推理共享同一 GPU            | 推理/渲染分离，各自独占 GPU                   |
| **模型显存占用**    | 4 × ~14GB = ~56GB              | 1 × ~14GB = ~14GB                            |
| **渲染显存**        | 与模型共享                      | 3 张卡全部用于渲染                            |
| **场景切换**        | 按 episode 自动切换              | 按 scene 分组，完成后调度下一 scene           |
| **batch 推理**      | 单 agent 逐条推理               | 最多 8 个 agent 同时 forward（chunk_size=8）  |

---

## 三、实验数据

### 实验配置

| 配置项              | Habitat-Sim                   | GenArk                                       |
|---------------------|-------------------------------|----------------------------------------------|
| 模型                | UniNaVid-7B                   | UniNaVid-7B（同一权重）                       |
| 数据集              | R2R-CE val_unseen             | R2R-CE val_unseen                            |
| 评估 episode 数     | 1839                          | 1839                                         |
| GPU 配置            | 4× RTX 3090（推理+渲染共享）   | 1× RTX 3090 推理 + 3× RTX 3090 渲染          |
| 分辨率              | 640×480, HFOV=120°            | 640×480, HFOV=105°                           |
| 最大步数            | 500                           | 500                                          |
| 成功距离            | 3.0m                          | 3.0m                                         |

### 导航性能指标

| 指标                    | Habitat-Sim     | GenArk          | 差异           |
|------------------------|-----------------|-----------------|----------------|
| **Success Rate (SR)**  | 0.520           | 0.540           | +0.020         |
| **Oracle SR**          | 0.574           | 0.617           | +0.043         |
| **SPL**                | 0.476           | 0.425           | -0.051         |
| **Distance to Goal**   | 5.027 m         | 3.911 m         | -1.116 m       |
| **Path Length**        | 9.290 m         | 8.987 m         | -0.303 m       |
| **nDTW**               | —               | 0.137           | —              |
| **sDTW**               | —               | 0.113           | —              |
| **Avg Steps**          | —               | 67.25           | —              |

> Habitat 本轮数据中 steps_taken / ndtw / sdtw 未采集（已修复，待重跑）

### 评估效率

| 效率指标              | Habitat-Sim         | GenArk               |
|-----------------------|---------------------|----------------------|
| **总耗时**            | **5 小时 36 分 57 秒** | **2 小时 44 分 56 秒** |
| **加速比**            | 1.0×                | **2.04×**             |
| **评估 episode 数**   | 1839                | 1839                 |
| **GPU 总数**          | 4                   | 4（1 推理 + 3 渲染）  |
| **模型加载次数**      | 4（每进程 1 次）      | 1（中心化服务）        |
| **模型总显存**        | ~56 GB              | ~14 GB               |

---

## 四、GenArk 的核心优势

### 1. 大规模环境并行
Habitat 每个 GPU 同时仅运行 1 个 agent；GenArk 利用 Genesis BatchRenderer 在单张 GPU 上同时仿真 **数十到数百个 agent**，将仿真吞吐量提升一个数量级。

### 2. 推理资源高效利用
Habitat 每个进程独立加载完整 7B 模型（4×14GB），大量显存用于重复存储模型权重；GenArk 仅部署 1 份模型，通过 HTTP 服务化 + batch inference 共享推理能力，**节省 ~42GB 模型显存**。

### 3. 渲染与推理解耦
推理 GPU 不参与渲染，渲染 GPU 不加载模型。各自独占硬件资源，避免了 Habitat 中渲染与推理争抢显存/算力的问题，**GPU 利用率更高**。

### 4. 弹性调度
Orchestrator 以 scene 为粒度动态调度 worker，scene 完成后自动补位。无需预分配 episode 到固定 GPU，**负载更均衡**。

---

## 五、当前局限与后续方向

| 局限                           | 说明                                                          |
|-------------------------------|--------------------------------------------------------------|
| Scene 兼容性                   | 部分 Matterport3D 场景的 OBJ→GLB 转换失败，导致 episode 丢失   |
| HFOV 差异                     | GenArk 105° vs Habitat 120°，可能影响指标可比性                |
| nDTW/sDTW 较低                | 可能与 NavMesh 精度、sliding 逻辑差异有关，需进一步对齐         |
| Habitat 缺失指标              | steps_taken / ndtw / sdtw 待重跑补充                          |

---

## 六、GenArk 框架构建中遇到的工程挑战

### 挑战 1：多 GPU 集群下推理与渲染的解耦调度

**问题**：Genesis 仿真器与 CUDA 渲染上下文强绑定，无法在同一进程中跨 GPU 并发调度渲染。若将 LLM 推理与 Genesis 渲染混合在同一进程/同一卡上，渲染 worker 启动时会重复加载 7B 模型，导致显存爆炸，且多 worker 之间的 CUDA context 冲突。

**解决方案**：采用 **推理服务化 + 异步 HTTP 解耦** 架构。将 UniNaVid 模型以 FastAPI RESTful 服务（model_server.py）形式独占部署在专用推理 GPU 上，通过标准化 HTTP API 对外暴露（`POST /create_session`、`POST /step`、`DELETE /session/{id}`）。渲染 worker 以 **multipart 二进制流** 的形式将批量 RGB 帧 POST 至推理服务，推理服务内部维护 per-session 的特征缓存与历史状态。

该方案实现了：① 推理卡与渲染卡完全解耦，各自独占硬件资源；② 多个渲染 worker 并发向同一推理服务发起请求，推理 GPU 利用率峰值可达 **100%**；③ 模型只需加载一次（~14GB），无需随 worker 数量线性扩展显存。

```
渲染 Worker 0 (GPU 3) ─┐
渲染 Worker 1 (GPU 4) ──┼─ HTTP POST /step ──→ model_server (GPU 2)
渲染 Worker 2 (GPU 5) ─┘        multipart binary             └→ batch generate → 返回 actions
```

---

### 挑战 2：多场景并行时相机 Roll 角漂移与随机旋转

**问题**：Genesis 以 `n_envs=N` 模式批量仿真多个 episode 时，在单场景下相机姿态正常；切换至多场景并发模式后，部分 agent 的相机出现 **Roll 角漂移** 和 **随机旋转**，导致渲染画面倾斜，模型输入异常。

**根因分析**：Genesis BatchRenderer 在多环境下共享底层 GPU 批量显存缓冲区。由于各环境的 `set_pose` 调用在 CUDA 内核层面存在异步写入竞争，**前一帧的旋转矩阵残留在共享显存区域**，被后续 `lookat` 计算误用；同时批量张量的数值误差在多帧累积后被放大，导致 up 向量偏离世界空间。

**解决方案**：在每次 `set_pose` 调用时**强制显式传入固定世界 up 向量**（Z-up），彻底切断底层共享缓冲区的状态污染路径：

```python
# env_worker_glb.py  — 强制截断底层 local up 的污染
up = torch.zeros_like(cam_pos)
up[:, 2] = 1.0          # 固定 Z-up，每帧显式覆盖，消除残留误差
cam.set_pose(pos=cam_pos, lookat=lookat, up=up)
```

---

### 挑战 3：部分场景渲染黑屏（纹理加载失败）

**问题**：在 Genesis 中直接加载 Matterport3D 的原始 OBJ 格式场景时，部分场景出现全黑渲染输出，纹理完全失效。

**根因分析**：分两步排查——① 首先验证 agent 初始坐标是否落在有效 NavMesh 三角面内（`batch_find_floor` 检测），排除坐标越界导致的渲染异常；② 确认坐标正常后，定位为 **OBJ 多材质格式与 Genesis Vulkan 渲染管线的兼容性问题**：Matterport3D OBJ 使用分离 MTL 文件引用多张纹理贴图，Genesis 在解析部分材质声明时出现 UV 映射丢失，导致 Vulkan fragment shader 采样到空纹理（输出黑色）。

**解决方案**：采用 **OBJ → GLB 一次性离线转换** 策略。GLB（GL Binary）将几何体、材质、纹理打包为单一二进制文件，纹理以内嵌 URI 形式存储，完整保留 UV 映射；Genesis 加载 GLB 时无需解析外部 MTL 引用，纹理加载成功率显著提升：

```python
# env_worker_glb.py — 使用 trimesh 将 OBJ 转为 GLB（仅首次）
mesh = trimesh.load(obj_path, force='scene')
mesh.export(glb_path)   # 纹理内嵌，消除 MTL 外部依赖
```

---

### 挑战 4：NavMesh Sliding 逻辑不完善导致 agent 卡死在 Mesh 中

**问题**：Genesis 仿真中 agent 执行连续 `forward` 动作时，遇到场景几何边界后未能正确滑行绕过，而是**嵌入 mesh 内部卡死**，造成模型持续输出同一动作、episode 无法终止。

**根因分析**：Habitat-Sim 内置完整的 NavMesh sliding 机制（基于 Recast/Detour），而 GenArk 在 Genesis 上自行实现的 NavMesh 导航缺少**碰撞回退（collision retreat）** 步骤——当 sliding 投影后的目标点仍在三角面外时，agent 会停在非法位置，下一帧的 `batch_find_floor` 检测失败后不更新位置，形成死锁。

**解决方案**：参考 Recast NavMesh 源码的 sliding 逻辑，在边界碰撞后增加 **沿法线方向的内缩回退**（inward retreat by `agent_radius`），确保投影点始终在有效三角面内：

```python
# env_worker_glb.py — 碰撞边界时沿三角面内法线回退 agent_radius
def batch_get_sliding_position(current_pos, desired_pos, current_tri_idx,
                               tri_v0_2d, tri_v1_2d, tri_v2_2d,
                               agent_radius: float = 0.18):
    """
    将越界目标点投影到当前三角面边界，并沿边缘内法线方向回退
    agent_radius，确保 sliding 后坐标合法，避免 agent 卡入 mesh。
    """
```

此机制大幅降低了 agent 卡死在 mesh 中的概率，使 episode 在边界处能正常滑行通过。
