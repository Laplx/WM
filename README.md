# WM

本仓库提供 WC2、WC2 高效版（WC2-st）和 W40 + 独立 canon 三种配置的训练代码与 RoboTwin 仿真参考推理入口，供真机团队对接。真机侧的 ROS 节点、相机同步和关节执行器仍需按实验室设备实现。

## 与出成绩的代码对齐

三种配置的推理与 RoboTwin 2.0 官方全量（clean 50 / 任务）出成绩时的代码逐位一致：

| 模式 | 配置 | 官方全量 Easy / Hard / Avg | 出成绩时的代码 |
|---|---|---|---|
| `wc2` | WC2 WM + 三个腕部 IDM | 84.04 / 61.98 / 73.01 | 研究仓库 5bcbb68 |
| `wc2-fast` | 同上 + `--idm_stale obs`（WC2-st） | 开发全量 86.33 / 67.33（官方全量在跑） | 研究仓库 bea81e1 |
| `w40-canon` | W40 WM + 独立 canon + 三个腕部 IDM | 暂定 84.73 / 60.62 / 72.67（Easy 缺 2 个任务在补） | 研究仓库 dd8f3e8 |

核对方法：同一组输入帧与本体、每次决策前同一随机种子，分别用出成绩时的代码和本仓库构建策略，连续 10 次决策（含中途换新回合）的动作块逐位相同；本仓库另开 `--graphs --idm_graph --idm_ms` 也逐位相同。之后的研究仓库改动在这三条路径上都只加了默认关闭的选项。

## 运行参考策略

在 Linux CUDA 工作站安装与本机驱动兼容的 PyTorch、下列 Python 依赖和 RoboTwin 2.0 的独立仿真环境。仿真入口从 `ROBOTWIN_ROOT` 加载 RoboTwin。

```bash
python -m pip install -r requirements.txt
export ROBOTWIN_ROOT=/path/to/RoboTwin
export WMRUNS=/path/to/checkpoints
export WMDATA=/path/to/inference-data
export PROBWM_WEIGHTS=/path/to/encoder-models
bash scripts/run_robotwin.sh wc2 --tasks handover_block --cfg demo_clean --n_ep 1 --n_env 1 --out runs/wc2.json
FAST=1 bash scripts/run_robotwin.sh wc2-fast --tasks handover_block --cfg demo_clean --n_ep 1 --n_env 1 --out runs/wc2st.json
```

模式换成 `wc2-fast` 或 `w40-canon` 即可复用相同评测设置。`FAST=1` 另开三个只改速度、不改结果的选项（见下文"时延"）。WC2 的两个模式共享全部参数文件，只有高效版启用 `--idm_stale obs`。W40 使用自己的 WM 与独立 canon。三者共用三个腕部 IDM。权重单独分发。参考加载器还使用 `WMDATA` 下的 `feats/dinov3_vits16hwf/info.json`、`meta/tasks.json`、`text/flan-t5-large_seen64.pt`。预训练编码器为 `facebook/dinov3-vits16-pretrain-lvd1689m` 和 `google/flan-t5-large`。

## 每次决策做什么

参数与出成绩时相同：`--sampler scratch4 --roll 1,0,8 --cfg_w 3 --fsteps 2 --ensemble 0.01 --exec 2 --jpeg std`。

1. 编码：头部与左、右腕三路图像过冻结的 DINOv3-S；W40 模式下头部特征再过独立 canon。
2. WM（滚动去噪）：`--roll 1,0,8` 表示每次决策只走 1 个 Euler 步、不额外回噪；某个时距槽这次没有对应时刻的估计时，借用 8 帧以内最近的已有估计。主线 WM 带稀疏远槽，走多尺度滚动路径，**没有 Langevin 步**。语言 CFG 3 时条件 / 无条件两路合成一个 batch 算。新回合的第一次决策没有上一次的状态，改用 `scratch4` 从纯噪声完整采样 4 步，比平时慢。
3. IDM：三个腕部 IDM 读同一份计划、历史和本体，各采一个动作块（flow 头 2 步），在标准化空间等权平均。
4. 执行：每次决策执行动作块的前 2 个路点（`--exec 2`），动作块之间按 `--ensemble 0.01` 做时间集成。
5. 高效版（`--idm_stale obs`）：IDM 读上一次决策留下的干净计划，WM 这一步挪到动作发出之后做，不在"帧 → 动作"的关键路径上。

## 时延

服务器 GPU（NVIDIA B20Z，B200 级）独占、B=1、开 `--graphs --idm_graph --idm_ms` 时的每次决策耗时（毫秒，p50）：

| 模式 | 关键路径（帧 → 动作） | 其中编码 / WM / IDM | 每次决策总计算 |
|---|---|---|---|
| wc2-fast | 12.4 | 3.8 / 22.5（在动作之后）/ 7.1 | 34.3 |
| wc2 | 33.8 | 3.7 / 22.1 / 7.1 | 33.8 |
| w40-canon | 39.8 | 3.7 + canon 7.2 / 21.2 / 7.2 | 39.8 |

- `--graphs`：编码器、WM、IDM 速度场、canon 的前向按输入形状捕获 CUDA graph；`--idm_graph`：IDM 骨干也捕获；`--idm_ms`：三个 IDM 各走一条 CUDA stream 并发（三者顺序算时高效版关键路径为 17.2）。三者都只改速度，策略输出逐位不变。
- WM 每次决策只走 1 步，在 B=1 时已经受算力限制（B=6 的 WM 耗时约为 B=1 的 5.3 倍），换到算力较低的卡上会按算力比例变慢，是主要瓶颈。高效版只把 WM 移出关键路径，WM 仍须在下一次决策之前算完。
- 在自己的卡上实测：`python scripts/latency.py ... --graphs --idm_graph --idm_ms --prof --B 1 --out lat.json`（参数见脚本开头），`--prof` 给出编码 / canon / WM / IDM 分项。

## 真机节奏

- 两次决策之间的实际时间 = `--exec`（路点数）× WM 训练数据的帧间隔。WM 的历史帧与滚动缓存都按"每 2 个路点决策一次"训练，推理时不要单独改 `--exec`；需要更长的决策间隔时，应在训练时加大取帧间隔。
- 不卡顿的条件：每次决策的全部计算（高效版包括动作之后的 WM 一步）小于决策间隔；关键路径最好明显小于一个路点的时长，否则动作块到达时前几个路点已过期，需要按时间戳跳过。
- 每个新回合单独重置历史、上一次计划和动作集成状态。

## 训练和真机对接

`python -m probwm.encode --help` 给出头部、左腕、右腕图像与指令的编码参数。`python -m probwm.train --help` 训练 WM，`python -m probwm.idm train --help` 训练 IDM。独立 canon 通过 WM 训练入口的 `--canon` 配对数据选项训练。已有带特征库的 WM 可用 `--init` 初始化。训练需要演示、特征、指令与对应动作。仿真配方中的数据及任务划分应按实际实验指定。

`probwm.robotwin_eval` 展示三路图像的顺序（头、左腕、右腕）、头部与腕部的归一化、指令编码、14 维关节状态、按帧更新的历史和动作块集成。真机接入时请先离线回放，核对关节顺序、夹爪量程和时间戳。机旁接口应将带时标的策略输出交给本机执行器。

本发布支持上面的三种参考配置。共用源码中保留了其他研究开关；启用这些开关可能需要完整研究仓库。
