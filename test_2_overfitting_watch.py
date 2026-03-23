import json
import os
import argparse
from glob import glob

def run_overfitting_watch(run_dir):
    print("="*60)
    print("TEST 2: The 'Overfitting' Watcher")
    print("="*60)

    metrics_file = os.path.join(run_dir, "valid_metrics.json")
    if not os.path.exists(metrics_file):
        print(f"⚠️  Metrics file not found at {metrics_file}.")
        print("Run training for at least 1 epoch to generate valid_metrics.json.")
        return

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    best_epoch = metrics.get('epoch', 'N/A')
    best_f1 = metrics.get('f1 propeptides', 0.0)

    print(f"Current Best Epoch: {best_epoch}")
    print(f"Best Validation F1: {best_f1:.4f}")

    if isinstance(best_epoch, int):
        if best_epoch < 10:
            print("⚠️ WARNING: The loss hasn't moved past epoch 10. The Focal Loss weight might be too aggressive, or the LR is too high.")
        elif best_epoch > 23:
            print("✅ SUCCESS: The model surpassed the V4 peak of Epoch 23. Weight Decay and Dropout are effectively preventing early memorization!")
        else:
            print("⏳ NEUTRAL: The model is peaking around Epoch 23 like V4. If training finishes here, consider bumping regularization further.")

    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Watch for Overfitting in V5 Run")
    parser.add_argument("--run_dir", type=str, default="esm3_propeptide_v5", help="Directory of the training run")
    args = parser.parse_args()

    run_overfitting_watch(args.run_dir)
