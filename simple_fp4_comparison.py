#!/usr/bin/env python3
"""
Minimal FP4 vs Baseline Training Comparison
Run: python simple_fp4_comparison.py
"""
import subprocess
import json

def run_and_extract_metrics(script):
    result = subprocess.run(['python', script], capture_output=True, text=True)
    lines = result.stdout.split('\n')
    for line in lines:
        if 'Validation Loss:' in line: loss = float(line.split(':')[1].strip())
        if 'Validation Accuracy:' in line: acc = float(line.split(':')[1].strip())
    return loss, acc

print("🔬 FP4 vs Baseline Comparison")
print("Running baseline model...")
base_loss, base_acc = run_and_extract_metrics('llm_base.py')

print("Running FP4 model...")
fp4_loss, fp4_acc = run_and_extract_metrics('llm_weights_fp4.py')

print(f"\n📊 Results:")
print(f"Baseline:  Loss={base_loss:.3f}, Acc={base_acc:.1%}")
print(f"FP4:       Loss={fp4_loss:.3f}, Acc={fp4_acc:.1%}")
print(f"Difference: Loss={fp4_loss-base_loss:+.3f}, Acc={fp4_acc-base_acc:+.1%}")
print(f"\n💡 Conclusion: FP4 quantization {'works' if fp4_acc > 0.5 else 'fails'} for training")