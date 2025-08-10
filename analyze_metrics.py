#!/usr/bin/env python3
"""
Script to analyze and visualize training metrics from JSON files.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_metrics(filename):
    """Load metrics from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return None

def plot_training_curves(baseline_data, fp4_data):
    """Plot training curves comparing baseline and FP4 models"""
    
    if not baseline_data or not fp4_data:
        print("❌ Missing data for plotting")
        return
    
    baseline_checkpoints = baseline_data.get('checkpoints', [])
    fp4_checkpoints = fp4_data.get('checkpoints', [])
    
    if not baseline_checkpoints or not fp4_checkpoints:
        print("❌ No checkpoint data available")
        return
    
    # Extract data for plotting
    baseline_steps = [cp['step'] for cp in baseline_checkpoints]
    baseline_losses = [cp['val_loss'] for cp in baseline_checkpoints]
    baseline_accs = [cp['val_accuracy'] for cp in baseline_checkpoints]
    
    fp4_steps = [cp['step'] for cp in fp4_checkpoints]
    fp4_losses = [cp['val_loss'] for cp in fp4_checkpoints]
    fp4_accs = [cp['val_accuracy'] for cp in fp4_checkpoints]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Comparison: Baseline (BF16) vs FP4 Quantized', fontsize=16)
    
    # Validation Loss
    ax1.plot(baseline_steps, baseline_losses, 'b-', label='Baseline (BF16)', linewidth=2)
    ax1.plot(fp4_steps, fp4_losses, 'r-', label='FP4 Quantized', linewidth=2)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation Accuracy
    ax2.plot(baseline_steps, baseline_accs, 'b-', label='Baseline (BF16)', linewidth=2)
    ax2.plot(fp4_steps, fp4_accs, 'r-', label='FP4 Quantized', linewidth=2)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss Difference (FP4 - Baseline)
    common_steps = sorted(set(baseline_steps) & set(fp4_steps))
    baseline_by_step = {cp['step']: cp for cp in baseline_checkpoints}
    fp4_by_step = {cp['step']: cp for cp in fp4_checkpoints}
    
    loss_diffs = []
    acc_diffs = []
    
    for step in common_steps:
        loss_diff = fp4_by_step[step]['val_loss'] - baseline_by_step[step]['val_loss']
        acc_diff = fp4_by_step[step]['val_accuracy'] - baseline_by_step[step]['val_accuracy']
        loss_diffs.append(loss_diff)
        acc_diffs.append(acc_diff)
    
    ax3.plot(common_steps, loss_diffs, 'g-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Loss Difference (FP4 - Baseline)')
    ax3.set_title('Validation Loss Difference')
    ax3.grid(True, alpha=0.3)
    
    # Accuracy Difference (FP4 - Baseline)
    ax4.plot(common_steps, acc_diffs, 'purple', linewidth=2)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Accuracy Difference (FP4 - Baseline)')
    ax4.set_title('Validation Accuracy Difference')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Training curves saved as 'training_comparison.png'")

def print_detailed_analysis(baseline_data, fp4_data):
    """Print detailed analysis of the training metrics"""
    
    print(f"\n{'='*80}")
    print("🔍 DETAILED TRAINING ANALYSIS")
    print(f"{'='*80}")
    
    if not baseline_data or not fp4_data:
        print("❌ Missing data for analysis")
        return
    
    baseline_checkpoints = baseline_data.get('checkpoints', [])
    fp4_checkpoints = fp4_data.get('checkpoints', [])
    
    if not baseline_checkpoints or not fp4_checkpoints:
        print("❌ No checkpoint data available")
        return
    
    # Model configurations
    print(f"\n📋 MODEL CONFIGURATIONS:")
    baseline_config = baseline_data.get('config', {})
    fp4_config = fp4_data.get('config', {})
    
    print(f"Architecture: {baseline_config.get('d_model')}d, {baseline_config.get('n_layers')}L, {baseline_config.get('n_heads')}H")
    print(f"Baseline LR: {baseline_config.get('adamw_lr', 'N/A')}")
    print(f"FP4 LR: {fp4_config.get('adamw_lr', 'N/A')}")
    
    # Training progress analysis
    print(f"\n📈 TRAINING PROGRESS:")
    
    # Find common checkpoints
    baseline_by_step = {cp['step']: cp for cp in baseline_checkpoints}
    fp4_by_step = {cp['step']: cp for cp in fp4_checkpoints}
    common_steps = sorted(set(baseline_by_step.keys()) & set(fp4_by_step.keys()))
    
    if common_steps:
        print(f"{'Step':<8} {'Base Loss':<12} {'FP4 Loss':<12} {'Loss Δ':<10} {'Base Acc':<12} {'FP4 Acc':<12} {'Acc Δ':<10}")
        print("-" * 80)
        
        for step in common_steps:
            base_cp = baseline_by_step[step]
            fp4_cp = fp4_by_step[step]
            
            loss_diff = fp4_cp['val_loss'] - base_cp['val_loss']
            acc_diff = fp4_cp['val_accuracy'] - base_cp['val_accuracy']
            
            print(f"{step:<8} {base_cp['val_loss']:<12.4f} {fp4_cp['val_loss']:<12.4f} {loss_diff:<+10.4f} "
                  f"{base_cp['val_accuracy']:<12.4f} {fp4_cp['val_accuracy']:<12.4f} {acc_diff:<+10.4f}")
    
    # FP4 quantization error analysis
    if 'checkpoints' in fp4_data:
        fp4_errors = [cp.get('fp4_quantization_error', 0) for cp in fp4_checkpoints if 'fp4_quantization_error' in cp]
        if fp4_errors:
            print(f"\n🔢 FP4 QUANTIZATION ERROR ANALYSIS:")
            print(f"Mean error: {np.mean(fp4_errors):.6f}")
            print(f"Max error: {np.max(fp4_errors):.6f}")
            print(f"Min error: {np.min(fp4_errors):.6f}")
            print(f"Std error: {np.std(fp4_errors):.6f}")
    
    # Training time analysis
    baseline_time = baseline_data.get('total_training_time', 0)
    fp4_time = fp4_data.get('total_training_time', 0)
    
    if baseline_time and fp4_time:
        time_diff = fp4_time - baseline_time
        time_pct = (time_diff / baseline_time) * 100
        print(f"\n⏱️ TRAINING TIME ANALYSIS:")
        print(f"Baseline time: {baseline_time:.1f} seconds ({baseline_time/60:.1f} minutes)")
        print(f"FP4 time: {fp4_time:.1f} seconds ({fp4_time/60:.1f} minutes)")
        print(f"Time difference: {time_diff:+.1f} seconds ({time_pct:+.1f}%)")

def main():
    """Main analysis function"""
    print("📊 Training Metrics Analysis")
    
    # Load metrics files
    baseline_data = load_metrics('baseline_training_metrics.json')
    fp4_data = load_metrics('fp4_training_metrics.json')
    
    if not baseline_data and not fp4_data:
        print("❌ No metrics files found. Run the training scripts first.")
        return
    
    # Print detailed analysis
    print_detailed_analysis(baseline_data, fp4_data)
    
    # Plot training curves
    try:
        plot_training_curves(baseline_data, fp4_data)
    except ImportError:
        print("⚠️ matplotlib not available, skipping plots")
    except Exception as e:
        print(f"⚠️ Error creating plots: {e}")

if __name__ == "__main__":
    main()