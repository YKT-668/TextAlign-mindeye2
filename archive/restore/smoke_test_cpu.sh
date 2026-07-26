#!/bin/bash
# smoke_test_cpu.sh - CPU-only smoke test for ConceptAlign restored environment
echo "=== ConceptAlign CPU Smoke Test ==="
python3 -c "
import sys; print('Python:', sys.version)
import torch; print('torch:', torch.__version__)
import transformers; print('transformers:', transformers.__version__)
import diffusers; print('diffusers:', diffusers.__version__)
import open_clip; print('open_clip: OK')
import numpy; print('numpy:', numpy.__version__)
import pandas; print('pandas:', pandas.__version__)
import scipy; print('scipy:', scipy.__version__)
import h5py; print('h5py:', h5py.__version__)
import sqlite3; print('sqlite3: OK')
import huggingface_hub; print('huggingface_hub:', huggingface_hub.__version__)
print('[PASS] All imports successful')
" 2>&1 || echo "[FAIL] Import check"
echo "[OK] Smoke test complete"