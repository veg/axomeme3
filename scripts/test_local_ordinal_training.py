#!/usr/bin/env python3
import os, sys, torch

PROJECT_ROOT = "/Users/sergei/Projects/TOGA_MEME"
sys.path.insert(0, PROJECT_ROOT)

from scripts.train_transformer_selection import train_full_model

print("=" * 90)
print("🚀 RUNNING LOCAL TEST OF AXOMEME 2.0 (6-BIN ORDINAL TRANSFORMER + SOFT EXPECTATION)")
print("=" * 90)

db_path = os.path.join(PROJECT_ROOT, "meme_results.db")
msa_dir = os.path.join(PROJECT_ROOT, "msa")

train_full_model(
    db_path=db_path,
    msa_dir=msa_dir,
    epochs=2,
    batch_size=256,
    micro_batch_size=64,
    lr=2e-4,
    subsample_limit=5000,
    subsample_val_limit=1024,
    epoch_size_limit=4096,
    loss_type="coral",
    device_override="cpu",
    save_path="test_axomeme_2.0_ordinal_checkpoint.pt",
    embed_dim=128,
    num_layers=2,
    num_heads=4
)
