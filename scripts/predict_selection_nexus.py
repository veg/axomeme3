#!/usr/bin/env python3
"""
predict_selection_nexus.py
--------------------------
Driver script to load a trained 5-head PhyloAxialTransformer selection model,
convert a NEXUS alignment and a NEXUS/Newick tree into model inputs,
and run codon-level selection predictions.

Usage:
  python3 scripts/predict_selection_nexus.py \
      --alignment msa/A1BG.gz \
      --model selection_transformer_best.pt \
      --output A1BG_selection_predictions.csv
"""

import os
import sys

# Forward directly to updated predict_multitask_nexus.py driver
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
multitask_script = os.path.join(SCRIPT_DIR, "predict_multitask_nexus.py")

if __name__ == "__main__":
    import subprocess
    cmd = [sys.executable, multitask_script] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
