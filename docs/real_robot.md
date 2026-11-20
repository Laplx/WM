# 真机数据准备与微调

本文说明怎样把 COBOT MAGIC 双臂真机示范整理成本仓库能直接训练的格式，并从仿真权重微调 WM 与腕部 IDM。命令与参数取自训练仿真权重时的配置；真机数据上的整条流程还没有跑通过，第一次请先用几条轨迹走一遍，核对每一步的输出。

## 0. 先确认的信息

- 轨迹条数与时长；单任务时每条轨迹对应的语言指令。
- 动作记录的是关节位置还是末端位姿（见第 2 节）。
- 相机：头部、左腕、右腕三路是否都有，分辨率，每帧时间戳。
- 关节与夹爪的单位、顺序、量程。

## 1. 时间结构

仿真里关节路点 16.7 Hz，WM 每 2 个路点取 1 帧（8.3 Hz）；每次决策执行 2 个路点，正好是 1 个 WM 帧。远槽 t+16、24、32、48、64 帧约对应 2–8 秒。

30 Hz 真机数据建议：

| 方案 | 路点频率 | WM 帧率 | 决策间隔 | 远槽覆盖 | 说明 |
|---|---|---|---|---|---|
| 推荐 | 15 Hz（每 2 个采样取 1） | 7.5 Hz | 133 ms | 约 2.1–8.5 s | 最接近仿真的时间尺度，时延预算也最宽 |
| 备选 | 20 Hz（按时间戳重采样） | 10 Hz | 100 ms | 约 1.6–6.4 s | |
| 不建议 | 30 Hz | 15 Hz | 67 ms | 约 1.1–4.3 s | 时间尺度只有仿真的一半左右，时延预算很紧 |

HDF5 里存的是路点频率的序列（见第 3 节），建根时 `--stride 2` 再取一半作为 WM 帧。部署时每次决策的全部计算须在决策间隔内完成（用 `scripts/latency.py` 实测）。

### 降频档（时延超过 133 ms 时）

每次决策的全部计算放不进 133 ms（15 Hz 路点、每 2 个路点决策一次）时，改用降频档：每 4 个路点决策一次，WM 的 1 帧也改成 4 个路点。

| 档位 | 路点频率 | WM 帧率 | 每次决策执行 | 决策间隔 | 远槽覆盖 |
|---|---|---|---|---|---|
| 默认 | 15 Hz | 7.5 Hz | 2 个路点 | 133 ms | 约 2.1–8.5 s |
| 降频档 | 15 Hz | 3.75 Hz | 4 个路点 | 约 267 ms | 约 4.3–17 s |

- 路点频率与动作分辨率不变，只是 WM 帧之间隔 4 个路点，每次决策执行 4 个路点（= 1 个 WM 帧），滚动窗口每次平移 1 帧。动作块仍是 16 个路点。
- **必须按新节奏重训 WM 和 IDM**：WM 加 `--delta 2`（在同一个数据根上隔一帧取 WM 帧，不用重新建根或编码），IDM 加 `--wpf 4 --H 4`（帧—路点比 4，存进 checkpoint，推理时自动读取）。不能把现有模型的 `--exec` 直接改成 4：WM 的历史帧间隔、滚动缓存的平移量和 IDM 读的计划帧都是按 2 个路点一帧训练的。训练命令见第 5、6 节末尾。
- 推理：`--exec 4`，其余参数不变。仿真参考入口：`bash scripts/run_robotwin.sh w40-canon-d4 ...`（W40 + 独立 canon + 单个降频档 IDM）。直接调用 `Policy` 时另传 `dec_wpf=4`（`robotwin_eval` 按 IDM checkpoint 自动传）；`--exec` 与 IDM 的帧—路点比不一致、或 WM 不是 `--delta 2` 训的，会直接报错。
- 仿真结论：每 4 个路点决策一次与每 2 个路点一次表现相当；每 8 个路点一次明显变差，不要再降。
- 降频档兼容 `FAST=1`（输出与不开时逐位相同）、`COMPILE=1`、`--chunk_sg`；不与第 9 节的异步执行（`--sim_delay`）和训练时 RTC（`--rtc_dmax` / `--rtc`）同用（没有验证过，代码里会报错）。降频档的决策间隔约 267 ms，按同步执行使用：每次决策的全部计算须在 267 ms 内完成。

## 2. 动作与本体

- 14 维，顺序同 RoboTwin aloha-agilex：左臂 6 个关节角（弧度）、左夹爪、右臂 6 个关节角、右夹爪。
- 每个路点存当时的关节位置 `q_t`。建根时自动生成动作 `a_t = q_{t+1}`（下一个路点的关节位置，末帧保持），本体为 `q_t`。
- 夹爪值映射到仿真数据的量程（对照 RoboTwin 官方数据集 `joint_action/vector` 的第 7、14 维），开合方向一致。
- 只有末端位姿时：IDM 输出的是关节位置，先用机械臂的逆运动学把末端位姿转成关节角，再按上面的格式存。不建议首轮改成末端空间的 IDM，那会偏离仿真权重。
- 末端位姿另存在 `endpose`（只作记录，训练不用）；没有就填 0。

## 3. 文件格式（RoboTwin legacy HDF5）

```
<真机根>/<任务名>/data/episode<k>.hdf5          k = 0, 1, …, N−1（连续编号）
    joint_action/vector        float32 (T, 14)   路点频率上的关节位置
    endpose/left_endpose       float32 (T, 7)    xyz + 四元数（可全 0）
    endpose/right_endpose      float32 (T, 7)
    observation/head_camera/rgb    (T,) 变长字节   每个元素 = cv2.imencode(".jpg", rgb)[1]（直接编 RGB 数组，不转 BGR）
    observation/left_camera/rgb    (T,)
    observation/right_camera/rgb   (T,)
<真机根>/<任务名>/instructions/episode<k>.json  {"seen": ["指令 1", …], "unseen": []}
```

- 三路图像与关节位置逐帧对齐（同一时刻，按时间戳取最近帧）。分辨率任意，建根与编码时按 `to256` 非等比缩放到 256×256（仿真是 320×240 拉伸），与推理时相同。
- 划分：建根时 episode 编号 ≥ `--val_from`（默认 45）的记为验证集。按条数设，留出约 10% 作验证，例如 50 条时 `--val_from 45`；用第 4 节的图像增强副本时取脚本打印的值。

## 4. 建根与编码

```bash
D=/path/to/real_root_built          # 输出的数据根
python -m probwm.render_robotwin legacy --out $D --clean_root /path/to/真机根 --tasks <任务名> --P 4 --stride 2 --workers 8 --val_from 45
for k in rgb wl wr; do               # 头部、左腕、右腕，都是 16×16 全分辨率（不加 --pool）
  python -m probwm.encode --data $D --key $k --alloc --parts 1
  python -m probwm.encode --data $D --key $k --src_h5 --shard 0 --nshard 1 --workers 8 --bs 256
done
python -m probwm.encode --data $D --text --text_model google/flan-t5-large --per_demo_aug 4 --tag seen64 --text_maxlen 64
mkdir -p $D/feats/dinov3_vits16hwf
cp $WMDATA/feats/dinov3_vits16hwf/info.json $D/feats/dinov3_vits16hwf/   # 三视图（头部 + 左腕 + 右腕）的视图定义
```

得到 `meta/`（索引、动作、本体、指令）、`h5/`（路点频率的完整动作，给 IDM）、`feats/`（三路特征与三视图定义）、`text/flan-t5-large_seen64.pt`。

### 可选：离线图像增强副本

仿真权重训练时只在特征 / 条件层面加了扰动（WM 的文本嵌入噪声 `--txt_noise 0.3` 与 10% 语言丢弃、IDM 的输入特征噪声 `--noise 0.1`，上面的命令已包含），没有做图像层面的增强；仿真的"随机化"数据是仿真器直接生成的场景，真机上没有对应物。真机数据少、画面单一，可以在建根之前给每条训练 episode 生成 K 份图像增强副本：

```bash
python scripts/aug_legacy.py --src /path/to/真机根 --dst /path/to/增强根 --tasks <任务名> --k 4 --n_val 5 --workers 8
# 最后一行打印 "--val_from N"；建根时把 --clean_root 换成增强根、--val_from 换成 N，其余命令不变
```

- 每份副本、每路相机抽一组参数，整条 episode 内所有帧相同（WM 要预测未来帧的特征，逐帧随机会让它去学预测抖动）：亮度 / 对比度 / 饱和度 ±20%、色调 ±3%，放大 0–8%、平移 ±5%、旋转 ±3°（反射填边）。`--no_geo` 只做颜色，`--geo_wrist 0` 腕部只做颜色。参数记在增强根的 `<任务>/aug_map.json`。
- 验证 episode 原样放在最后、不增强；动作、本体、末端位姿原样复制，不做镜像这类会改变关节含义的增强。
- 作用：对光照、相机位姿的小偏差、轻微遮挡更鲁棒；同样步数下每条原始示范被重复的轮数变少，过拟合也少一些。它改变不了物体之间的相对位置，应对摆放变化仍要靠示范本身覆盖更多初始位置。
- 代价：K = 4 时训练集特征约为原来的 5 倍（35 分钟数据约 40 GB）。这一项没有在仿真里验证过，建议做 / 不做各训一版，比较验证损失与第 8 节的离线回放。

## 5. 微调 WM（从 WC2 出发）

```bash
W=$WMRUNS/rteasy/wmX2F5Clw3WC2/ckpt.pt
python -m probwm.train --data $D --suites '' --text flan-t5-large_seen64 --mode abs --K 4 --roll 0.5 --roll_sort --hw 0.5 \
  --far_clamp 8 --alpha 13 --txt_noise 0.3 --p_main 0 --seed 0 --encoder dinov3_vits16hwf --tgt_tok 256 --xin_zero --xtok_last \
  --d 1280 --depth 12 --heads 20 --n_fut 14 --tgt_offs 0,1,2,3,4,5,6,7,8,16,24,32,48,64 --canon_slot --lw_motion 3 --lw_slot \
  --init $W --init_offs --norm_from $W --lr 3e-5 --warmup 200 --steps 4000 --bs 8 --accum 8 \
  --log 100 --save 500 --keep 1000,1500,2000,2500,3000,3500 --out runs/wm_real
```

- 以上与训练 WC2 时的配置逐项相同，只改数据根、步数、日志 / 存点间隔和批的切法（全局批 64 = `--bs 8 --accum 8`）。`--norm_from` 沿用仿真权重的特征归一化。
- 热启动一律用小学习率（3e-5）、短 warmup；大学习率会把模型踢出原来的极小点。不加 `--trainval`，保留验证集。
- 存下的是 EMA 权重（`--ema 0.999`，时间常数约 1000 步）；`--keep` 另存 `ckpt_s<步>.pt`，最后一步存 `ckpt.pt`。

**训多少步**：按"轮数"定，不按固定步数。1 轮 ≈ 训练集 WM 帧数 ÷ 64 步；单任务、从仿真权重热启动时 10–20 轮通常够，再多容易背下示范的布局和轨迹（仿真里续训越久，语言和改写指令越差）。

例：50 条示范、共约 35 分钟，30 Hz 录制、按 15 Hz 存路点（第 1 节）：
- WM 帧 ≈ 35 × 60 × 7.5 ≈ 1.6 万；`--val_from 45` 留 5 条验证后，训练集约 1.4 万帧，1 轮 ≈ 220 步。
- 不做图像增强：`--steps 4000`（约 18 轮），每 500 步存一份，同上面的命令。
- 做图像增强（K = 4）：训练集约 7 万帧，1 轮 ≈ 1100 步；`--steps 6000 --keep 2000,3000,4000,5000`。

**挑哪一份**：日志每 `--log` 步打印一次 `val(ema)`（单个验证批，噪声大），取相邻 5 次的平均看走势，选平均值不再下降的最早那一份（通常在 2–3k 步）；再对它和后一份各做一次第 8 节的离线回放，选回放更好的。若到最后一份验证损失仍在明显下降，把 `--steps` 改到约 6000 重训（不增强时）。
- 真机数据没有仿真里的随机化配对，`--canon_slot` 的 t+0 目标就是该帧本身。
- 也可以从 W40（`wmX2F5Clw3W`）出发：去掉 `--canon_slot`、`--tgt_offs` 和 `--n_fut` 改成 W40 的 13 槽（`1,…,8,16,24,32,48,64`）。独立 canon 是在仿真随机化配对上训的，真机画面上没有对应的配对数据，首轮建议用 WC2。

**小显存卡（如 24 GB）**：保持全局批 64 不变，只用 `--accum` 换小的微批：`--bs 1 --accum 64`，并设环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 实测训练阶段的显存峰值（上面的 WC2 配置，不开 `--compile`，含验证）：微批 1 约 21 GB，2 约 26–28 GB，4 约 38 GB。24 GB 卡上只放得下微批 1，余量很小，卡上不要有别的进程；仍放不下就换显存更大的卡。训练代码没有梯度检查点一类的省显存开关。
- 微批越小每步越慢；步数不变（轮数按全局批算）。
- 不要直接缩小全局批。真要缩小（如 `--bs 4 --accum 4`，全局批 16）：学习率保持 3e-5，不放大；步数按"轮数"同比放大（1 轮 = 训练集 WM 帧数 ÷ 全局批，全局批从 64 降到 16 时步数 × 4）。

**降频档（第 1 节）的 WM**：从仿真降频档权重 `$WMRUNS/dec4/wmD4/ckpt.pt` 出发（W40 加 `--delta 2` 在仿真数据上按同样的小学习率续训 20k 步得到；W40 的 13 槽、没有 t+0 槽），命令同上面的 W40 写法，另加 `--delta 2`：

```bash
W=$WMRUNS/dec4/wmD4/ckpt.pt
python -m probwm.train --data $D --suites '' --text flan-t5-large_seen64 --mode abs --K 4 --delta 2 --roll 0.5 --roll_sort --hw 0.5 \
  --far_clamp 8 --alpha 13 --txt_noise 0.3 --p_main 0 --seed 0 --encoder dinov3_vits16hwf --tgt_tok 256 --xin_zero --xtok_last \
  --d 1280 --depth 12 --heads 20 --n_fut 13 --tgt_offs 1,2,3,4,5,6,7,8,16,24,32,48,64 --lw_motion 3 --lw_slot \
  --init $W --norm_from $W --lr 3e-5 --warmup 200 --steps 4000 --bs 8 --accum 8 \
  --log 100 --save 500 --keep 1000,1500,2000,2500,3000,3500 --out runs/wm_real_d4
```

- 数据根与默认档相同（`--stride 2` 建根），`--delta 2` 在训练时隔一帧取 WM 帧；奇偶两种相位的帧都作起点，只是每条示范末尾约 16 个数据帧不作起点。步数按默认档的算法估（1 轮 ≈ 训练集 WM 帧数 ÷ 64 步），挑选方法同上。
- 推理时配独立 canon（与仿真降频档相同，同 `w40-canon` 模式）；真机上 canon 没有对应的配对数据，若担心可先不开 canon 对比离线回放。

## 6. 微调腕部 IDM

```bash
for s in 0 1 2; do
  I=$WMRUNS/rteasy/idmPLwr$([ $s = 0 ] || echo s$s).pt
  PROBWM_FEATS_MMAP=1 python -m probwm.idm train --kind idm --ih 3 --d 768 --depth 6 --H 8 --head flow --proprio qpos \
    --data $D --h5 $D/h5 --wm $WMRUNS/rt/c/oursC/wm512/ckpt.pt --wrist 1 --wkeys wl,wr --seed $s \
    --init $I --lr 3e-5 --steps 1500 --log 100 --out runs/idm_real_s$s.pt
done
```

- 结构参数与仿真的三个腕部 IDM 相同（`torch.load(ckpt)["args"]` 可查全部），`--init` 从对应种子出发。`--wm` 只用来取头部特征的编码名与归一化（数据根由 `--data` 给）：须是只读头部特征的 WM，用仿真的 `wm512`（训练三个腕部 IDM 时也是它）；它的头部归一化与 W40 / WC2 及第 5 节微调后的 WM 相同（都沿用 `--norm_from`），与推理时一致。不要指向第 5 节微调出来的 WM：它读三视图特征（`dinov3_vits16hwf`），IDM 会去找不存在的 `dinov3_vits16hwf_wl` 而报错。
- 先只适配并验证种子 0 的单个 IDM；效果正常再训另两个，推理时用 `--idm … --idm_ens …` 做三者集成。
- 步数：IDM 批 256，1 轮 ≈ 训练集 WM 帧数 ÷ 256 步。上面的例子里 1 轮约 55 步，`--steps 1500` 约 27 轮；做图像增强时用 `--steps 2500`。IDM 只在结束时存一份，日志里 `val nmse` 若在最后几百步回升，减少步数重训（单次只要十几分钟）。输入特征噪声 `--noise 0.1` 是默认值，不要关。
- 核对每个关节和夹爪的语义、量程与仿真一致。

**小显存卡（如 24 GB）**：保持全局批 256，用 `--bs 256 --accum 8`（每次前向 32 个），另加 `--val_chunk 128`（验证分块）和 `--wstat_cpu`（腕部特征的归一化统计量在 CPU 上算），并设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- `--wstat_cpu` 不开时，启动阶段统计腕部归一化要多占约 30 GB 显存，与批大小无关；小显存卡上不开会在启动时就显存不足。结果与在 GPU 上算只差浮点舍入。
- 实测训练阶段的显存峰值（`--val_chunk 128`，含验证）：每次前向 16 个约 18 GB，32 个约 20–22 GB，64 个约 23 GB。从小往大试，能放下就用大的（更快）；32 个放不下就改 `--accum 16`。
- `--accum k` 与不分块的整批在数学上相同（关掉 bf16 后梯度只差浮点累加顺序），学习率与步数都不用改。全局批真要缩小时同第 5 节：学习率不放大，步数按轮数同比放大。

**降频档（第 1 节）的 IDM**：从仿真降频档 IDM `$WMRUNS/dec4/idmD4.pt` 出发（`--wpf 4 --H 4`：每帧 4 个路点、读计划的前 4 帧、动作块仍 16 个路点；结构参数同上）。`--wm` 给只读头部特征的 WM（仿真的 `wm512`，它只提供头部特征的编码名与归一化，和训练 `idmD4` 时一样；`--data` 换成真机数据根，帧间隔由 `--wpf` 决定）：

```bash
PROBWM_FEATS_MMAP=1 python -m probwm.idm train --kind idm --ih 3 --d 768 --depth 6 --H 4 --wpf 4 --head flow --proprio qpos \
  --data $D --h5 $D/h5 --wm $WMRUNS/rt/c/oursC/wm512/ckpt.pt --wrist 1 --wkeys wl,wr --seed 0 \
  --init $WMRUNS/dec4/idmD4.pt --lr 3e-5 --steps 1500 --log 100 --out runs/idm_real_d4.pt
```

- 1 轮的步数同默认档（训练集 WM 帧数 ÷ 256），步数与判断方法同上。
- 默认档的 IDM（`--H 8`，帧—路点比 2）不能作降频档 IDM 的初始化：输入帧数不同，结构对不上。
- 仿真里降频档只训了一个 IDM（种子 0），闭环也是单 IDM 测的。真机上要集成时，可再用 `--seed 1`、`--seed 2` 从同一个 `idmD4` 出发各训一个，推理加 `--idm_ens`；这种集成没有在仿真里验证过。
- 不与 `--rtc_dmax` 同用（代码里会报错）。

## 7. 动作平滑与抖动

仿真里测过发出的关节目标：相邻动作块交界处没有系统性跳变，不平滑主要来自 IDM 每个路点的采样噪声，靠下面两项压住。真机出现抖动时先核对这几点：

- **时间集成必须开**：`--ensemble 0.01`（相邻约 8 个动作块近似等权平均）。仿真闭环里关掉它，成功率大幅下降，换块处出现大跳变，夹爪在换块处来回开合：相邻动作块本来就分歧很大（包括开合时刻），是时间集成把分歧抹平的。真机出现明显抖动时先核对这一项是否开着；不要缩短集成窗口，也不要调大系数，两者都会更抖。
- 用三个 IDM 等权集成（`--idm ... --idm_ens ...`）；单个 IDM 明显更抖。
- 优先同步执行（每次决策完成后再发下一段）。若必须异步执行，按时间戳对齐、丢掉新块里已过期的路点：仿真里延迟 2 个路点时成功率不变，但换块处的跳变会变大，更依赖下一条的平滑。训练时 RTC（IDM 加"已发出动作前缀"输入，见第 9 节）能把这部分跳变收回大半，但没有提高成功率，不建议默认使用。
- 块内 SG 平滑（可选，零延迟）：`--chunk_sg 7,2`（直接用 `Policy` 时传 `chunk_sg="7,2"`）。IDM 的动作块（三个种子平均、解码到原始单位后）在进时间集成之前，沿块内 16 个路点做中心 7 点二次 Savitzky–Golay 拟合（两端用首 / 末 7 个点拟合），只平滑 12 个臂关节，两个夹爪维不动；集成缓存里存的是平滑后的块。只用这一块自己的路点，不增加延迟。仿真里它明显降低逐路点的采样抖动、关节来回反向和空闲臂的目标游走，不伤成功率。空闲或慢速段有细小的高频抖动 / 嗡鸣、或位置环较刚时开。只和时间集成一起用；不与第 9 节的 `--rtc` 同用（会改动前缀位）。
- RoboTwin 仿真对每个路点单独做轨迹时间参数化，会吸收指令里的小幅抖动；真机若用固定频率的刚性位置环，路点之间需要插值或平滑，否则这些小幅抖动会直接变成振动。
- 空闲的那只臂，目标值会有很小的持续游走；必要时可在机旁加死区（连续几步运动量低于噪声底就保持不动）。这一项没有闭环验证过。

## 8. 上机前

1. 离线回放：拿一条留出的轨迹，逐帧画出输入时刻、WM 预测和 IDM 动作对照标签，先排除错一帧、错一只手、关节顺序或夹爪方向不对。
2. 影子推理：先用录好的观测，再用在线相机与关节反馈，只记录不执行，核对节奏（第 1 节）和每个新回合的状态重置。
3. 推理配置同 README（`--sampler scratch4 --roll 1,0,8 --cfg_w 3 --fsteps 2 --ensemble 0.01 --exec 2 --jpeg std`，WC2-st 另加 `--idm_stale obs`；降频档改 `--exec 4`），把 WM / IDM 换成微调后的权重，`--wm_data` 指向真机数据根（读其中的特征定义、任务表与指令表）。
4. 再按 README 的"真机调试顺序"与第 7 节排查时延与抖动；改成异步执行时见第 9 节。

## 9. 异步执行与 RTC

默认同步执行（每次决策算完再发下一段）加时间集成，用不到本节。只有改成异步执行（边执行旧动作块边算新块）时才需要。

**先算 d。** 实测从拍帧到新动作块可以执行的端到端延迟 p95（相机 + 预处理 + 网络 + 推理；`scripts/latency.py` 只测推理部分），d = ⌈延迟 / 路点周期⌉。
- d ≤ 1：不需要处理。
- d = 2–4：先按时间戳对齐、丢掉新块里已过期的前 d 个路点（仿真里的对应选项是 `--sim_delay d`）。仿真里这样做不伤成功率；只有换块处出现明显抖动、且按第 7 节排查后仍在，才用训练时 RTC。
- d > 4：超出 RTC 训练时采样的延迟范围（0–4 个路点）。先降延迟（机旁本地卡、`FAST=1`、`COMPILE=1`、减少传输），或改用第 1 节的降频档（同步执行，决策间隔约 267 ms）。

**训练时 RTC 是什么。** IDM 多学一个条件：动作块的前 d 个路点是已经下发、改不了的"前缀"，网络接着前缀生成后面的路点。训练时每个样本随机取 d ∈ {0, …, 4}（d = 0 占 1/3），前缀位置用真值、不加噪，只在其余位置算损失。结构只多一个零初始化的前缀嵌入；不给前缀时用法和输出都与原 IDM 相同。仿真里它不伤同步执行；延迟 2 个路点时，成功率与"直接丢过期路点"相同，换块处的跳变明显变小。

**训练。** 和第 6 节的真机微调合成一次，只多两个参数，步数同第 6 节：

```bash
for s in 0 1 2; do
  I=$WMRUNS/rtc/idmPLwrR$([ $s = 0 ] || echo s$s).pt   # 仿真 RTC 权重（若随权重分发）；没有就用第 6 节的 $WMRUNS/rteasy/idmPLwr….pt
  PROBWM_FEATS_MMAP=1 python -m probwm.idm train --kind idm --ih 3 --d 768 --depth 6 --H 8 --head flow --proprio qpos \
    --data $D --h5 $D/h5 --wm $WMRUNS/rt/c/oursC/wm512/ckpt.pt --wrist 1 --wkeys wl,wr --seed $s \
    --init $I --lr 3e-5 --steps 1500 --log 100 --rtc_dmax 4 --rtc_p0 0.333 --out runs/idm_real_rtc_s$s.pt
done
```

- `--init` 优先用仿真 RTC 权重（`idmPLwrR` / `idmPLwrRs1` / `idmPLwrRs2`，与 `idmPLwr` / `s1` / `s2` 同结构，只多一个前缀嵌入）。没有时从 `idmPLwr` 出发，前缀嵌入零初始化，日志打印"缺 pemb，零初始化"。
- 从 RTC 权重热启动时必须带 `--rtc_dmax`，否则报错。
- 日志多一列 `val[2:4]`：第 2–3 个路点在不给前缀与给真值前缀（d = 2）时的误差，后者应不高于前者。
- 推理时三个 IDM 都要是 RTC 训的（集成成员共用同一前缀）。
- 小显存卡同第 6 节（`--accum`、`--val_chunk`、`--wstat_cpu` 都可与 `--rtc_dmax` 同用；分块时损失仍按整批的非前缀路点数归一，与整批相同）。

**推理接口。** 每次决策：

```python
pol.rtc_prefix(P, dv)   # act 前设
acts = pol.act(...)     # acts[b, :dv[b]] 就是 P[b, :dv[b]]；机器人从第 dv[b] 个路点接着执行
pol.rtc_prefix(None)    # act 后清掉
```

- `P`：形状 (B, d, 14)，原始单位（同 `act` 的输出：关节角为弧度，夹爪同动作约定），内容是拍下这一帧之后、新动作块生效之前机器人还会执行的已下发路点，按时间先后排。B 与 `act` 的批大小、样本顺序相同（真机 B = 1）。numpy 或 torch 张量都可以。
- `dv`：(B,) 整数，各样本的有效前缀长度（≤ `P.shape[1]`）。新回合的第一次决策还没有已下发路点，给 0。`dv` 全为 0 或 `P=None` 时等价于不开，输出与原 IDM 逐位相同。
- d 按上面的端到端延迟 p95 取，一次运行内固定。执行队列：新块的前 d 个就是队列里正在执行的已下发路点，从第 d 个开始接上；之后每次决策照常执行 2 个路点，队列里始终留 d 个已下发路点给下一次推理期间执行。时间集成（`--ensemble`）照常开，新旧块按绝对路点时刻对齐，已下发的路点不再改。`probwm/robotwin_eval.py` 里 `--sim_delay` 的这段逻辑可以直接照抄到机旁节点。
- 与 `FAST=1`（`--graphs --idm_graph --idm_ms`）、`COMPILE=1`（`--compile`）都兼容。带前缀时输出的前缀位与 P 只差浮点舍入；`FAST=1` 下带不带前缀，输出都与 eager 逐位相同。带前缀时 flow 头的调用形状不同，`COMPILE=1` 会另编译一次，开机预热时带前缀、不带前缀各空跑几次决策。不与 `--idm_ens_par`、`--act_roll` 同用。

**上真机前先在仿真里对照。** 用同一个 d 各跑一遍"丢过期路点"和"前缀"：

```bash
FAST=1 bash scripts/run_robotwin.sh wc2-fast --sim_delay 2 --tasks <任务> --cfg demo_clean --n_ep 20 --out runs/skip.json
RTC=1 FAST=1 bash scripts/run_robotwin.sh wc2-fast --sim_delay 2 --rtc --tasks <任务> --cfg demo_clean --n_ep 20 --out runs/rtc.json
```

`--sim_delay d` 模拟推理期间照上一块再执行 d 个路点、新块从第 d 位接上；加 `--rtc` 把这 d 个路点作前缀；`RTC=1` 换成 RTC 训的三个 IDM（`$WMRUNS/rtc/idmPLwrR*.pt`）。`--sim_delay 0`（默认）就是同步执行。
