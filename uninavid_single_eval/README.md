# UniNaVid 单 Episode 推理

在 Genesis 仿真环境里运行 uni-navid 模型,对单条 episode 推理并导出 mp4。

## 文件
- `eval_single_episode.py` — 入口脚本,加载场景/navmesh/相机,循环调用 agent 并写视频
- `agent_uninavid_genesis.py` — UniNaVid agent (模型推理 forward/left/right/stop)
- `genesis_gt_path_multi_eval.py` — navmesh 寻地与 sliding 工具函数

## 依赖路径(已硬编码,如需修改改 `eval_single_episode.py` 顶部常量)
- EPISODES_FILE: `/ssd/hongyuding/code/Genesis/scripts/OpenNav_R2R-CE_100_bertidx.json`
- SCENE_DATASETS: `/ssd/hongyuding/data/genesis`
- MODEL_PATH: `model_zoo/uninavid-7b-full-224-video-fps-1-grid-2` (相对 NaVid-VLN-CE)

## 运行
```bash
cd /ssd/hongyuding/code/Genesis/scripts/NaVid-VLN-CE
conda activate genesis
python eval_single_episode.py --episode-id 42
```

可选参数:
- `--episode-id` 默认 42
- `--max-steps` 默认 250
- `--output` 默认 `output_uninavid_single/episode_42.mp4`
- `--fps` 默认 10

## 说明
动作来自 UniNaVid 模型推理(非 GT)。Agent 接收当前 RGB 与指令文本,生成
forward/left/right/stop,经 navmesh sliding 后更新相机位姿。
