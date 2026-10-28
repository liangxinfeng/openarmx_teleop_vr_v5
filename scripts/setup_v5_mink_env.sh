#!/usr/bin/env bash
set -euo pipefail
WS="${ROS2_WS:-/home/ubuntu/ros2_ws}"
BASE="$WS/venvs/openarmx_v4_placo"
DEST="$WS/venvs/openarmx_v5_mink"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -x "$BASE/bin/python" ]; then
  echo "Native Pinocchio environment is missing: $BASE" >&2; exit 2
fi
if [ ! -x "$DEST/bin/python" ]; then "$BASE/bin/python" -m venv "$DEST"; fi
"$DEST/bin/python" - "$BASE" <<'PY'
import sys,sysconfig
from pathlib import Path
base=Path(sys.argv[1])/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
assert base.is_dir(),base
p=Path(sysconfig.get_paths()['purelib'])/'v5_robot_native.pth'
p.write_text('import site; site.addsitedir('+repr(str(base))+')\n/usr/lib/python3/dist-packages\n')
PY
"$DEST/bin/python" -m pip install -r "$SCRIPT_DIR/../requirements-mink.txt"
"$DEST/bin/python" -c 'import mink,mujoco,pinocchio,quadprog,qpsolvers; assert "quadprog" in qpsolvers.available_solvers; print("Mink / quadprog native environment ready")'
