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
- 划分：建根时 episode 编号 ≥ 45 的记为验证集。数据少于约 50 条时，改 `probwm/render_robotwin.py` 里 `_legacy_task` 的 `split=int(k >= …)` 阈值，留出约 10% 的轨迹作验证。

## 4. 建根与编码

```bash
D=/path/to/real_root_built          # 输出的数据根
python -m probwm.render_robotwin legacy --out $D --clean_root /path/to/真机根 --tasks <任务名> --P 4 --stride 2 --workers 8
for k in rgb wl wr; do               # 头部、左腕、右腕，都是 16×16 全分辨率（不加 --pool）
  python -m probwm.encode --data $D --key $k --alloc --parts 1
  python -m probwm.encode --data $D --key $k --src_h5 --shard 0 --nshard 1 --workers 8 --bs 256
done
python -m probwm.encode --data $D --text --text_model google/flan-t5-large --per_demo_aug 4 --tag seen64 --text_maxlen 64
mkdir -p $D/feats/dinov3_vits16hwf
cp $WMDATA/feats/dinov3_vits16hwf/info.json $D/feats/dinov3_vits16hwf/   # 三视图（头部 + 左腕 + 右腕）的视图定义
```

得到 `meta/`（索引、动作、本体、指令）、`h5/`（路点频率的完整动作，给 IDM）、`feats/`（三路特征与三视图定义）、`text/flan-t5-large_seen64.pt`。

## 5. 微调 WM（从 WC2 出发）

```bash
W=$WMRUNS/rteasy/wmX2F5Clw3WC2/ckpt.pt
python -m probwm.train --data $D --suites '' --text flan-t5-large_seen64 --mode abs --K 4 --roll 0.5 --roll_sort --hw 0.5 \
  --far_clamp 8 --alpha 13 --txt_noise 0.3 --p_main 0 --seed 0 --encoder dinov3_vits16hwf --tgt_tok 256 --xin_zero --xtok_last \
  --d 1280 --depth 12 --heads 20 --n_fut 14 --tgt_offs 0,1,2,3,4,5,6,7,8,16,24,32,48,64 --canon_slot --lw_motion 3 --lw_slot \
  --init $W --init_offs --norm_from $W --lr 3e-5 --warmup 200 --steps 3000 --bs 8 --accum 8 --out runs/wm_real
```

- 以上与训练 WC2 时的配置逐项相同，只改数据根、步数和批的切法（全局批 64 = `--bs 8 --accum 8`）。`--norm_from` 沿用仿真权重的特征归一化。
- 热启动一律用小学习率（3e-5）、短 warmup；大学习率会把模型踢出原来的极小点。
- 步数按数据量定：单任务数据少，先训 2–5k 步，看验证损失和第 7 节的离线回放，再决定是否加训。不加 `--trainval`，保留验证集。
- 真机数据没有仿真里的随机化配对，`--canon_slot` 的 t+0 目标就是该帧本身。
- 也可以从 W40（`wmX2F5Clw3W`）出发：去掉 `--canon_slot`、`--tgt_offs` 和 `--n_fut` 改成 W40 的 13 槽（`1,…,8,16,24,32,48,64`）。独立 canon 是在仿真随机化配对上训的，真机画面上没有对应的配对数据，首轮建议用 WC2。

## 6. 微调腕部 IDM

```bash
for s in 0 1 2; do
  I=$WMRUNS/rteasy/idmPLwr$([ $s = 0 ] || echo s$s).pt
  PROBWM_FEATS_MMAP=1 python -m probwm.idm train --kind idm --ih 3 --d 768 --depth 6 --H 8 --head flow --proprio qpos \
    --data $D --h5 $D/h5 --wm runs/wm_real/ckpt.pt --wrist 1 --wkeys wl,wr --seed $s \
    --init $I --lr 3e-5 --steps 3000 --out runs/idm_real_s$s.pt
done
```

- 结构参数与仿真的三个腕部 IDM 相同（`torch.load(ckpt)["args"]` 可查全部），`--init` 从对应种子出发。`--wm` 只用来取数据根与头部特征的归一化，指向微调后的 WM，与推理时一致。
- 先只适配并验证种子 0 的单个 IDM；效果正常再训另两个，推理时用 `--idm … --idm_ens …` 做三者集成。
- 核对每个关节和夹爪的语义、量程与仿真一致。

## 7. 上机前

1. 离线回放：拿一条留出的轨迹，逐帧画出输入时刻、WM 预测和 IDM 动作对照标签，先排除错一帧、错一只手、关节顺序或夹爪方向不对。
2. 影子推理：先用录好的观测，再用在线相机与关节反馈，只记录不执行，核对节奏（第 1 节）和每个新回合的状态重置。
3. 推理配置同 README（`--sampler scratch4 --roll 1,0,8 --cfg_w 3 --fsteps 2 --ensemble 0.01 --exec 2 --jpeg std`，WC2-st 另加 `--idm_stale obs`），把 WM / IDM 换成微调后的权重，`--wm_data` 指向真机数据根（读其中的特征定义、任务表与指令表）。
4. 再按 README 的"真机调试顺序"排查时延与抖动。
