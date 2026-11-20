# WM

本仓库提供主线方法在真机上测试所需的推理与训练代码：三种推理配置（W40 + 独立 canon、WC2、WC2 高效版），WM / IDM 训练入口，以及 RoboTwin 仿真参考入口（用来核对输入处理与策略回路）。真机侧的 ROS 节点、相机同步和关节执行器需按实验室设备实现。

## 运行参考策略

在 Linux CUDA 工作站安装与本机驱动兼容的 PyTorch、下列 Python 依赖；仿真参考入口另需 RoboTwin 2.0 的独立环境，从 `ROBOTWIN_ROOT` 加载。

```bash
python -m pip install -r requirements.txt
export ROBOTWIN_ROOT=/path/to/RoboTwin
export WMRUNS=/path/to/checkpoints
export WMDATA=/path/to/inference-data
export PROBWM_WEIGHTS=/path/to/encoder-models
bash scripts/run_robotwin.sh w40-canon --tasks handover_block --cfg demo_clean --n_ep 1 --n_env 1 --out runs/w40.json
FAST=1 bash scripts/run_robotwin.sh wc2-fast --tasks handover_block --cfg demo_clean --n_ep 1 --n_env 1 --out runs/wc2st.json
```

- 模式：`w40-canon`（W40 WM + 独立 canon）、`wc2`（WC2 WM，规范化由 WM 的 t+0 槽承担）、`wc2-fast`（同 WC2，加 `--idm_stale obs` 的高效推理）。三者共用三个腕部 IDM（三个训练种子等权集成）。
- `FAST=1` 另开三个只改速度、不改策略输出的选项（见"时延"）。
- 权重单独分发。参考加载器还使用 `WMDATA` 下的 `feats/dinov3_vits16hwf/info.json`、`meta/tasks.json`、`text/flan-t5-large_seen64.pt`。预训练编码器为 `facebook/dinov3-vits16-pretrain-lvd1689m` 和 `google/flan-t5-large`。

## 每次决策做什么

推理参数：`--sampler scratch4 --roll 1,0,8 --cfg_w 3 --fsteps 2 --ensemble 0.01 --exec 2 --jpeg std`。

1. 编码：头部与左、右腕三路图像过冻结的 DINOv3-S；`w40-canon` 下头部特征再过独立 canon。
2. WM（滚动去噪）：`--roll 1,0,8` 表示每次决策只走 1 个 Euler 步、不额外回噪；某个时距槽这次没有对应时刻的估计时，借用 8 帧以内最近的已有估计。WM 带稀疏远槽，走多尺度滚动路径，没有 Langevin 步。语言 CFG 3 时条件 / 无条件两路合成一个 batch 计算。新回合的第一次决策没有上一次的状态，改用 `scratch4` 从纯噪声完整采样 4 步，比平时慢。
3. IDM：三个腕部 IDM 读同一份计划、历史和本体，各采一个动作块（flow 头 2 步），在标准化空间等权平均。
4. 执行：每次决策执行动作块的前 2 个路点（`--exec 2`），相邻动作块按 `--ensemble 0.01` 做时间集成。
5. 高效推理（`--idm_stale obs`）：IDM 读上一次决策留下的干净计划，WM 这一步挪到动作发出之后，不在"帧 → 动作"的关键路径上。

## 时延

- 提速选项：`--graphs`（编码器、WM、IDM 速度场、canon 的前向按输入形状捕获 CUDA graph）、`--idm_graph`（IDM 骨干也捕获）、`--idm_ms`（三个 IDM 各走一条 CUDA stream 并发）。三者只改速度，策略输出逐位不变。
- `--compile default`（`run_robotwin.sh` 用 `COMPILE=1`）：编码器、WM、canon、IDM 先过 torch.compile 再捕获 CUDA graph，每次决策的总计算约降到 0.5–0.65 倍（WM 约减半）。输出不逐位相同，差异与 bf16 数值噪声同量级，仿真闭环检验表现不变。首次决策要编译（冷启动约 100 s，`TORCHINDUCTOR_CACHE_DIR` 指向固定目录可复用缓存），每个新输入形状各编一次；真机 B=1 时开机先空跑几次决策预热。
- WM 每次决策只走 1 步，但在 B=1 时已受算力限制，是主要瓶颈；高效推理只把 WM 移出关键路径，WM 仍须在下一次决策之前算完。
- 在自己的卡上实测：`python scripts/latency.py ... --graphs --idm_graph --idm_ms --compile default --prof --B 1 --out lat.json`（参数见脚本开头）。`crit` 为帧 → 动作的关键路径，`total` 为每次决策的全部计算，`--prof` 给出编码 / canon / WM / IDM 分项。

## 真机节奏

- 时间结构：路点（动作）频率是 WM 帧率的 2 倍（仿真里路点 16.7 Hz、WM 帧 8.3 Hz），每次决策执行 2 个路点 = 1 个 WM 帧，窗口平移 1 帧。所以两次决策之间的实际时间 = WM 的 1 个帧间隔。WM 的历史帧与滚动缓存都按这个节奏训练，推理时不要单独改 `--exec`；需要更长的决策间隔时，应在训练时加大取帧间隔（路点同步按 2 倍帧率取）。
- 不卡顿的条件：每次决策的全部计算（高效推理包括动作之后的 WM 一步）小于决策间隔；关键路径最好明显小于一个路点的时长，否则动作块到达时前几个路点已过期，需按时间戳跳过。
- 每个新回合单独重置历史、上一次计划和动作集成状态。

## 真机调试顺序

1. 先在机旁的卡上用 `scripts/latency.py --prof` 实测，确认关键路径与每次决策的全部计算是否满足上面两条节奏条件。
2. 出现抖动时，先定位最早出现的层：同一关节画"预测动作块 → 发布目标 → 实测反馈"，一次只改一个因素。
3. WM 算不过来（每次决策的计算超过 1 个 WM 帧间隔）时的办法：训练时加大取帧间隔、保持 `--exec 2`，拉长决策间隔；或换更快的卡。关掉语言 CFG 能让 WM 计算约减半，但对效果的影响没有验证过。
4. RTC：默认同步执行加时间集成（`--ensemble 0.01`），不需要 RTC。改成异步执行（边执行旧动作块边算新块）时，按时间戳对齐、丢掉新块里已过期的路点即可：仿真里延迟 2 个路点时这样做表现不变。训练时 RTC（IDM 增加"已发出动作前缀"输入）在仿真里不伤同步执行，但延迟 2 个路点时没有带来改进，不建议默认使用（本仓库不含该功能）；它只让换块处的跳变变小，真机若异步后在换块处出现明显抖动再考虑。异步后的决策节奏也要与 WM 的取帧间隔、滚动缓存一致。

## 训练和真机对接

真机数据的格式、取帧频率、动作约定、建根编码与微调命令见 [docs/real_robot.md](docs/real_robot.md)。

`python -m probwm.encode --help` 给出头部、左腕、右腕图像与指令的编码参数。`python -m probwm.train --help` 训练 WM，`python -m probwm.idm train --help` 训练 IDM。已有带特征库的 WM 可用 `--init` 初始化；各权重的训练配置保存在 checkpoint 的 `args` 里，微调时沿用其中的视觉编码、历史长度、目标时距与模型尺寸。独立 canon 直接使用发布的权重。

`probwm.robotwin_eval` 展示三路图像的顺序（头、左腕、右腕）、图像预处理（`to256`、`--jpeg std`）、头部与腕部的归一化、指令编码、14 维关节状态、按帧更新的历史和动作块集成。真机接入时请先离线回放，核对关节顺序、夹爪量程和时间戳。机旁接口应将带时标的策略输出交给本机执行器。

本仓库只支持上面三种推理配置。共用源码中保留了其他研究开关，启用它们可能需要完整研究仓库。
