# WM

本仓库提供 WC2、WC2 高效版和 W40 + 独立 canon 的训练代码与 RoboTwin 仿真参考推理入口，供真机团队对接。真机侧的 ROS 节点、相机同步和关节执行器仍需按实验室设备实现。

## 运行参考策略

在 Linux CUDA 工作站安装与本机驱动兼容的 PyTorch、下列 Python 依赖和 RoboTwin 2.0 的独立仿真环境。训练或直接调用策略时按实际输入安装依赖。仿真入口从 `ROBOTWIN_ROOT` 加载 RoboTwin。

```bash
python -m pip install -r requirements.txt
export ROBOTWIN_ROOT=/path/to/RoboTwin
export WMRUNS=/path/to/checkpoints
export WMDATA=/path/to/inference-data
export PROBWM_WEIGHTS=/path/to/encoder-models
bash scripts/run_robotwin.sh wc2 --tasks handover_block --cfg demo_clean --n_ep 1 --n_env 1 --out runs/wc2.json
```

把模式换成 `wc2-fast` 或 `w40-canon` 即可复用相同评测设置。WC2 的两个模式共享全部参数文件，只有高效版启用 `--idm_stale obs`。W40 使用自己的 WM 与独立 canon。三者共用三个腕部 IDM。权重单独分发。参考加载器还使用 `WMDATA` 下的 `feats/dinov3_vits16hwf/info.json`、`meta/tasks.json`、`text/flan-t5-large_seen64.pt`。预训练编码器为 `facebook/dinov3-vits16-pretrain-lvd1689m` 和 `google/flan-t5-large`。

## 训练和真机对接

`python -m probwm.encode --help` 给出头部、左腕、右腕图像与指令的编码参数。`python -m probwm.train --help` 训练 WM，`python -m probwm.idm train --help` 训练 IDM。独立 canon 通过 WM 训练入口的 `--canon` 配对数据选项训练。已有带特征库的 WM 可用 `--init` 初始化。训练需要演示、特征、指令与对应动作。仿真配方中的数据及任务划分应按实际实验指定。

`probwm.robotwin_eval` 展示三路图像的顺序（头、左腕、右腕）、头部与腕部的归一化、指令编码、14 维关节状态、按帧更新的历史和动作块集成。真机接入时请先离线回放，核对关节顺序、夹爪量程和时间戳。机旁接口应将带时标的策略输出交给本机执行器。每个新回合单独重置历史、上一次计划和动作集成状态。

本发布支持上面的三种参考配置。共用源码中保留了其他研究开关；启用这些开关可能需要完整研究仓库。
