#!/bin/bash
# Test PolyInfer installation on Linux/macOS
# Usage: ./scripts/test_install.sh [extras]
#   ./scripts/test_install.sh          - Basic install
#   ./scripts/test_install.sh cpu      - Install with [cpu]
#   ./scripts/test_install.sh all      - Install with [all]

set -e

EXTRAS="$1"
VENV_DIR="test_venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "PolyInfer Installation Test (Linux/macOS)"
echo "============================================================"
echo "Project: $PROJECT_DIR"
echo "Extras: ${EXTRAS:-none}"
echo "============================================================"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -d "$VENV_DIR" ]; then
        rm -rf "$VENV_DIR"
        echo "Removed virtual environment"
    fi
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Remove existing venv
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv "$VENV_DIR"

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install polyinfer
if [ -z "$EXTRAS" ]; then
    echo "Installing polyinfer..."
    pip install -e "$PROJECT_DIR"
else
    echo "Installing polyinfer[$EXTRAS]..."
    pip install -e "$PROJECT_DIR[$EXTRAS]"
fi

# Verify imports
echo ""
echo "Verifying imports..."
python -c "
import polyinfer as pi
print(f'Version: {pi.__version__}')
print(f'Backends: {pi.list_backends()}')
print(f'Devices: {[d.name for d in pi.list_devices()]}')
"

# Install pytest and run tests
echo ""
echo "Running tests..."
pip install pytest -q
TEST_RESULT=0
python -m pytest "$PROJECT_DIR/tests" -v --tb=short -x --ignore=tests/test_benchmark.py || TEST_RESULT=$?

# Deactivate
deactivate 2>/dev/null || true

# Report result
echo ""
echo "============================================================"
if [ $TEST_RESULT -eq 0 ]; then
    echo "INSTALLATION TEST: PASSED"
else
    echo "INSTALLATION TEST: FAILED"
fi
echo "============================================================"

exit $TEST_RESULT
