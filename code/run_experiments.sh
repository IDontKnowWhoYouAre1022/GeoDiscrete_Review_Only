#!/bin/bash
# Quick start script for the directional faithfulness experiments.

set -e

echo "========================================"
echo "Directional Faithfulness Experiments"
echo "========================================"

if [ ! -f "run_all_experiments.py" ]; then
    echo "Error: run this script from code/exp"
    exit 1
fi

if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "Error: Python not found"
    exit 1
fi

echo "Python: $($PYTHON --version)"
echo "Checking NumPy..."
$PYTHON -c "import numpy; print('NumPy', numpy.__version__)"
echo "Checking plotting stack..."
$PYTHON -c "import matplotlib; import seaborn; print('Matplotlib', matplotlib.__version__); print('Seaborn', seaborn.__version__)"

MODE="full"
OUTPUT_DIR="./results"
EXP_NUM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODE="quick"
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --exp)
            EXP_NUM="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run_experiments.sh [--quick] [--output-dir DIR] [--exp N]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CMD="$PYTHON run_all_experiments.py --output-dir $OUTPUT_DIR"
if [ "$MODE" = "quick" ]; then
    CMD="$CMD --quick"
fi
if [ -n "$EXP_NUM" ]; then
    CMD="$CMD --exp $EXP_NUM"
fi

echo "Mode: $MODE"
echo "Output: $OUTPUT_DIR"
echo "Command: $CMD"
echo

$CMD

echo
echo "Experiments completed."
echo "Summaries: $OUTPUT_DIR/exp*/exp*_summary.txt"
echo "Figures:   $OUTPUT_DIR/exp*/*.pdf and $OUTPUT_DIR/exp*/*.png"
