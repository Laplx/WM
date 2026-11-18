#!/usr/bin/env bash
set -eu
: "${WMRUNS:?Set WMRUNS to the checkpoint root}"
: "${WMDATA:?Set WMDATA to the inference metadata root}"
: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to your RoboTwin checkout}"
case "${1:-}" in
  wc2)       mode=(--wm "$WMRUNS/rteasy/wmX2F5Clw3WC2/ckpt.pt") ;;
  wc2-fast)  mode=(--wm "$WMRUNS/rteasy/wmX2F5Clw3WC2/ckpt.pt" --idm_stale obs) ;;
  w40-canon) mode=(--wm "$WMRUNS/rteasy/wmX2F5Clw3W/ckpt.pt" --canon "$WMRUNS/rt/c/oursC/canon/ckpt.pt") ;;
  *) echo 'Usage: bash scripts/run_robotwin.sh {wc2|wc2-fast|w40-canon} [robotwin_eval flags]' >&2; exit 2 ;;
esac
shift
exec python -m probwm.robotwin_eval "${mode[@]}" \
  --wm_data "$WMDATA" --idm "$WMRUNS/rteasy/idmPLwr.pt" \
  --idm_ens "$WMRUNS/rteasy/idmPLwrs1.pt,$WMRUNS/rteasy/idmPLwrs2.pt" \
  --sampler scratch4 --roll 1,0,8 --cfg_w 3 --fsteps 2 --ensemble 0.01 --exec 2 --jpeg std "$@"
