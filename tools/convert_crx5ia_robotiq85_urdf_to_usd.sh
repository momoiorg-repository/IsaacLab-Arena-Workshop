#!/bin/bash
# Convert CRX5iA + Robotiq 85 URDF to USD using Isaac Sim's URDF converter.
#
# Joint parameters follow Isaac Sim Tutorial 6 recommendations:
#   - Arm joints (J1-J6): stiffness=400, damping=40
#   - finger_joint: stiffness=37.52, damping=0.00125 (set in ArticulationCfg)
#   - Mimic joints: need manual gearing in USD after import
#
# Run inside Isaac Sim Docker or an environment with IsaacLab installed:
#   bash tools/convert_crx5ia_robotiq85_urdf_to_usd.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

URDF_FILE="$REPO_ROOT/isaaclab_arena/embodiments/crx5ia/usd/crx5ia_robotiq85.urdf"
USD_OUTPUT_DIR="$REPO_ROOT/isaaclab_arena/embodiments/crx5ia/usd/crx5ia_robotiq85"
USD_OUTPUT="$USD_OUTPUT_DIR/crx5ia_robotiq85.usd"

if [ ! -f "$URDF_FILE" ]; then
    echo "ERROR: URDF not found at $URDF_FILE"
    echo "Run the following first (with ROS2 workspace sourced):"
    echo "  source ros2_ws/install/setup.bash"
    echo "  python tools/convert_crx5ia_to_urdf.py"
    echo ""
    echo "That script will generate crx5ia_robotiq85.urdf from crx5ia_robotiq85.urdf.xacro."
    exit 1
fi

mkdir -p "$USD_OUTPUT_DIR"

echo "=== Converting CRX5iA + Robotiq 85 URDF -> USD ==="
echo "  Input:  $URDF_FILE"
echo "  Output: $USD_OUTPUT"
echo ""

python "${REPO_ROOT}/submodules/IsaacLab/scripts/tools/convert_urdf.py" \
    "$URDF_FILE" \
    "$USD_OUTPUT" \
    --fix-base \
    --joint-stiffness 400.0 \
    --joint-damping 40.0 \
    --joint-target-type position \
    --headless

echo ""
echo "Done! USD written to: $USD_OUTPUT"
echo ""
echo "Next steps:"
echo "  1. Verify mimic joints are correctly configured in the USD."
echo "  2. Run: python tools/patch_usd_gravity.py $USD_OUTPUT"
echo "  3. Test: python tools/debug_env_cfg.py (update embodiment to crx5ia_robotiq85)"
