#!/usr/bin/env python3
"""
Comparison script to run both FP4 and baseline models for fair evaluation.
This script runs both models with the same configuration and compares their performance.
"""

import subprocess
import sys
import time
import json
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script and capture its output"""
    print(f"\n{'='*60}")
    print(f"🚀 Running {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Extract final metrics from output
        metrics = extract_metrics(result.stdout)
        metrics['duration'] = duration
        metrics['success'] = result.returncode == 0
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"❌ {description} timed out after 1 hour")
        return {'success': False, 'error': 'timeout'}
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return {'success': False, 'error': str(e)}

def extract_metrics(output):
    """Extract final metrics from script output"""
    metrics = {}
    
    lines = output.split('\n')
    for line in lines:
        if 'Validation Loss:' in line:
            try:
                metrics['val_loss'] = float(line.split('Validation Loss:')[1].strip())
            except:
                pass
        elif 'Validation Accuracy:' in line:
            try:
                metrics['val_accuracy'] = float(line.split('Validation Accuracy:')[1].strip())
            except:
                pass
        elif 'Validation Perplexity:' in line:
            try:
                metrics['val_perplexity'] = float(line.split('Validation Perplexity:')[1].strip())
            except:
                pass
        elif 'Total time:' in line and 'minutes' in line:
            try:
                time_str = line.split('Total time:')[1].split('minutes')[0].strip()
                metrics['training_time_minutes'] = float(time_str)
            except:
                pass
        elif 'Total parameters:' in line:
            try:
                params_str = line.split('Total parameters:')[1].strip()
                # Remove commas and extract number
                params_str = params_str.replace(',', '').split()[0]
                metrics['total_parameters'] = int(params_str)
            except:
                pass
        elif 'FP4 parameters:' in line:
            try:
                params_str = line.split('FP4 parameters:')[1].strip()
                params_str = params_str.replace(',', '').split()[0]
                metrics['fp4_parameters'] = int(params_str)
            except:
                pass
        elif 'Estimated memory savings:' in line:
            try:
                savings_str = line.split('Estimated memory savings: ~')[1].split('%')[0]
                metrics['memory_savings_percent'] = float(savings_str)
            except:
                pass
        elif 'Final FP4 quantization error:' in line:
            try:
                error_str = line.split('Final FP4 quantization error:')[1].strip()
                metrics['fp4_quantization_error'] = float(error_str)
            except:
                pass
    
    return metrics

def print_comparison(base_metrics, fp4_metrics):
    """Print a detailed comparison of the results"""
    print(f"\n{'='*80}")
    print("📊 FINAL COMPARISON RESULTS")
    print(f"{'='*80}")
    
    if not base_metrics.get('success', False):
        print("❌ Baseline model failed to complete")
        return
    
    if not fp4_metrics.get('success', False):
        print("❌ FP4 model failed to complete")
        return
    
    print(f"\n🏆 PERFORMANCE METRICS:")
    print(f"{'Metric':<25} {'Baseline (FP32)':<20} {'FP4 Quantized':<20} {'Difference':<15}")
    print("-" * 80)
    
    # Validation Loss (lower is better)
    if 'val_loss' in base_metrics and 'val_loss' in fp4_metrics:
        base_loss = base_metrics['val_loss']
        fp4_loss = fp4_metrics['val_loss']
        diff = fp4_loss - base_loss
        diff_pct = (diff / base_loss) * 100
        print(f"{'Validation Loss':<25} {base_loss:<20.4f} {fp4_loss:<20.4f} {diff:+.4f} ({diff_pct:+.1f}%)")
    
    # Validation Accuracy (higher is better)
    if 'val_accuracy' in base_metrics and 'val_accuracy' in fp4_metrics:
        base_acc = base_metrics['val_accuracy']
        fp4_acc = fp4_metrics['val_accuracy']
        diff = fp4_acc - base_acc
        diff_pct = (diff / base_acc) * 100
        print(f"{'Validation Accuracy':<25} {base_acc:<20.4f} {fp4_acc:<20.4f} {diff:+.4f} ({diff_pct:+.1f}%)")
    
    # Validation Perplexity (lower is better)
    if 'val_perplexity' in base_metrics and 'val_perplexity' in fp4_metrics:
        base_ppl = base_metrics['val_perplexity']
        fp4_ppl = fp4_metrics['val_perplexity']
        diff = fp4_ppl - base_ppl
        diff_pct = (diff / base_ppl) * 100
        print(f"{'Validation Perplexity':<25} {base_ppl:<20.2f} {fp4_ppl:<20.2f} {diff:+.2f} ({diff_pct:+.1f}%)")
    
    print(f"\n⏱️ TRAINING TIME:")
    if 'training_time_minutes' in base_metrics and 'training_time_minutes' in fp4_metrics:
        base_time = base_metrics['training_time_minutes']
        fp4_time = fp4_metrics['training_time_minutes']
        diff = fp4_time - base_time
        diff_pct = (diff / base_time) * 100
        print(f"{'Training Time (min)':<25} {base_time:<20.1f} {fp4_time:<20.1f} {diff:+.1f} ({diff_pct:+.1f}%)")
    
    print(f"\n💾 MEMORY EFFICIENCY:")
    if 'total_parameters' in fp4_metrics:
        total_params = fp4_metrics['total_parameters']
        print(f"{'Total Parameters':<25} {total_params:,}")
    
    if 'fp4_parameters' in fp4_metrics:
        fp4_params = fp4_metrics['fp4_parameters']
        total_params = fp4_metrics.get('total_parameters', fp4_params)
        fp4_ratio = (fp4_params / total_params) * 100
        print(f"{'FP4 Parameters':<25} {fp4_params:,} ({fp4_ratio:.1f}%)")
    
    if 'memory_savings_percent' in fp4_metrics:
        savings = fp4_metrics['memory_savings_percent']
        print(f"{'Est. Memory Savings':<25} ~{savings:.1f}%")
    
    if 'fp4_quantization_error' in fp4_metrics:
        error = fp4_metrics['fp4_quantization_error']
        print(f"{'FP4 Quantization Error':<25} {error:.6f}")
    
    print(f"\n🎯 SUMMARY:")
    
    # Determine winner based on validation loss
    if 'val_loss' in base_metrics and 'val_loss' in fp4_metrics:
        if fp4_metrics['val_loss'] < base_metrics['val_loss']:
            print("🏅 FP4 model achieved better validation loss!")
        elif fp4_metrics['val_loss'] > base_metrics['val_loss']:
            loss_degradation = ((fp4_metrics['val_loss'] - base_metrics['val_loss']) / base_metrics['val_loss']) * 100
            print(f"📉 FP4 model has {loss_degradation:.1f}% higher validation loss")
        else:
            print("🤝 Both models achieved similar validation loss")
    
    # Memory efficiency summary
    if 'memory_savings_percent' in fp4_metrics:
        savings = fp4_metrics['memory_savings_percent']
        print(f"💾 FP4 quantization provides ~{savings:.1f}% memory savings")
    
    # Training time comparison
    if 'training_time_minutes' in base_metrics and 'training_time_minutes' in fp4_metrics:
        if fp4_metrics['training_time_minutes'] < base_metrics['training_time_minutes']:
            time_savings = ((base_metrics['training_time_minutes'] - fp4_metrics['training_time_minutes']) / base_metrics['training_time_minutes']) * 100
            print(f"⚡ FP4 model trained {time_savings:.1f}% faster")
        else:
            time_overhead = ((fp4_metrics['training_time_minutes'] - base_metrics['training_time_minutes']) / base_metrics['training_time_minutes']) * 100
            print(f"⏳ FP4 model took {time_overhead:.1f}% longer to train")

def main():
    """Main comparison function"""
    print("🔬 LLM FP4 vs Baseline Comparison")
    print("This script will run both models and compare their performance.")
    
    # Check if both scripts exist
    base_script = Path("llm_base.py")
    fp4_script = Path("llm_weights_fp4.py")
    
    if not base_script.exists():
        print(f"❌ Baseline script not found: {base_script}")
        return
    
    if not fp4_script.exists():
        print(f"❌ FP4 script not found: {fp4_script}")
        return
    
    print(f"✅ Found both scripts")
    
    # Run baseline model
    print(f"\n🔄 Starting comparison...")
    base_metrics = run_script("llm_base.py", "Baseline Model (FP32 + AdamW)")
    
    # Run FP4 model
    fp4_metrics = run_script("llm_weights_fp4.py", "FP4 Quantized Model (FP4 + AdamW)")
    
    # Print comparison
    print_comparison(base_metrics, fp4_metrics)
    
    # Save results to JSON
    results = {
        'baseline': base_metrics,
        'fp4': fp4_metrics,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to comparison_results.json")

if __name__ == "__main__":
    main()