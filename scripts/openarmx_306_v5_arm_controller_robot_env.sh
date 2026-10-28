#!/usr/bin/env bash
set -e

V5_PYTHON="/home/ubuntu/ros2_ws/venvs/openarmx_v5_mink/bin/python"
if [ ! -x "$V5_PYTHON" ]; then
  echo "V5 Mink Python environment not found: $V5_PYTHON" >&2
  echo "Run scripts/setup_v5_mink_env.sh once, then rebuild this package." >&2
  exit 2
fi

exec "$V5_PYTHON" -m openarmx_teleop_vr_306_v5.controller_node "$@"
