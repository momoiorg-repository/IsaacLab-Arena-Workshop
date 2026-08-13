#!/bin/bash
# Convert CRX5iA + Robotiq 140 URDF to USD using Isaac Sim's URDF converter.
#
# Based on Isaac Sim Tutorial 6 "Setup a Manipulator" recommended parameters:
#   - Arm joints (J1-J6): Natural Frequency = 300, position target
#   - finger_joint: Stiffness = 37.52, Damping = 0.00125, Max Force = 1000
#   - Mimic joints: Natural Frequency = 2500, Damping Ratio = 0.005
#
# Note: Mimic joint gearing must be adjusted in the USD after import.
#
# Run inside Isaac Sim Docker or an environment with IsaacLab installed:
#   bash tools/convert_crx5ia_urdf_to_usd.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

URDF_FILE="$REPO_ROOT/isaaclab_arena/embodiments/crx5ia/usd/crx5ia.urdf"
USD_OUTPUT="$REPO_ROOT/isaaclab_arena/embodiments/crx5ia/usd/crx5ia.usd"

if [ ! -f "$URDF_FILE" ]; then
    echo "ERROR: URDF not found at $URDF_FILE"
    echo "Run 'python tools/convert_crx5ia_to_urdf.py' first."
    exit 1
fi

echo "=== Converting CRX5iA + Robotiq 140 URDF -> USD ==="
echo "  Input:  $URDF_FILE"
echo "  Output: $USD_OUTPUT"
echo ""
echo "  Joint parameters (per Isaac Sim Tutorial 6):"
echo "    Arm joints: stiffness=400, damping=40 (Natural Freq ~300)"
echo "    Robotiq finger_joint: mimic joints need manual gearing in USD"
echo ""

# Use IsaacLab's convert_urdf.py script
# --fix-base: creates a fixed joint from world to base_link
# Joint stiffness/damping are initial values; mimic joint gearing
# must be configured post-import in the USD or via ArticulationCfg.
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
echo "IMPORTANT: After conversion, verify mimic joints are correctly configured."
echo "See Isaac Sim Tutorial 6 for expected mimic joint parameters."
