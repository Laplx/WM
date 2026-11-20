#!/usr/bin/env bash
# 三种推理配置的 RoboTwin 仿真参考入口。默认参数即主线推理配置；FAST=1 另开三个提速选项
# （--graphs --idm_graph --idm_ms：CUDA graph + 三个 IDM 多流并发，Policy 输出与默认逐位相同，只改速度）；
# COMPILE=1 再加 --compile default（torch.compile，不逐位相同，差在 bf16 数值噪声量级）。
set -eu
: "${WMRUNS:?Set WMRUNS to the checkpoint root}"
: "${WMDATA:?Set WMDATA to the inference metadata root}"
: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to your RoboTwin checkout}"
case "${1:-}" in
  wc2)       mode=(--wm "$WMRUNS/rteasy/wmX2F5Clw3WC2/ckpt.pt") ;;
  wc2-fast)  mode=(--wm "$WMRUNS/rteasy/wmX2F5Clw3WC2/ckpt.pt" --idm_stale obs) ;;
  w40-canon) mode=(--wm "$WMRUNS/rteasy/wmX2F5Clw3W/ckpt.pt" --canon "$WMRUNS/rt/c/oursC/canon/ckpt.pt") ;;
  *) echo 'Usage: [FAST=1] [COMPILE=1] bash scripts/run_robotwin.sh {wc2|wc2-fast|w40-canon} [robotwin_eval flags]' >&2; exit 2 ;;
esac
shift
fast=()
[ "${FAST:-0}" = 1 ] && fast=(--graphs --idm_graph --idm_ms)
[ "${COMPILE:-0}" = 1 ] && fast+=(--compile default)
exec python -m probwm.robotwin_eval "${mode[@]}" ${fast[@]+"${fast[@]}"} \
  --wm_data "$WMDATA" --idm "$WMRUNS/rteasy/idmPLwr.pt" \
  --idm_ens "$WMRUNS/rteasy/idmPLwrs1.pt,$WMRUNS/rteasy/idmPLwrs2.pt" \
  --sampler scratch4 --roll 1,0,8 --cfg_w 3 --fsteps 2 --ensemble 0.01 --exec 2 --jpeg std "$@"
