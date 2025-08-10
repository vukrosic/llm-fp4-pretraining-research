import torch
import bitsandbytes.functional as bnb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set device and seed for reproducibility
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

print(f"FP4 Visual Analysis - Using device: {device}")
print("Generating comprehensive charts...")

# Create figure with subplots
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 16))

# ============================================================================
# Chart 1: Range Representation Capability
# ============================================================================
ax1 = plt.subplot(3, 3, 1)

exponents = np.arange(-8, 9, 1)
test_values = torch.tensor([10.0**exp for exp in exponents], dtype=torch.float32, device=device)
quantized, state = bnb.quantize_fp4(test_values)
dequantized = bnb.dequantize_fp4(quantized, state)

original_vals = test_values.cpu().numpy()
fp4_vals = dequantized.cpu().numpy()

ax1.semilogy(exponents, original_vals, 'o-', label='Original Float32', markersize=8, linewidth=2)
ax1.semilogy(exponents, fp4_vals, 's-', label='FP4 Quantized', markersize=8, linewidth=2)

# Highlight problematic regions
underflow_mask = fp4_vals == 0
overflow_mask = np.isinf(fp4_vals)

if np.any(underflow_mask):
    ax1.axvspan(exponents[underflow_mask].min() - 0.5, exponents[underflow_mask].max() + 0.5, 
                alpha=0.2, color='red', label='Underflow Region')

if np.any(overflow_mask):
    ax1.axvspan(exponents[overflow_mask].min() - 0.5, exponents[overflow_mask].max() + 0.5, 
                alpha=0.2, color='orange', label='Overflow Region')

ax1.set_xlabel('Exponent (10^x)')
ax1.set_ylabel('Value')
ax1.set_title('FP4 Range Representation Capability')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ============================================================================
# Chart 2: Precision Loss Across Different Scales
# ============================================================================
ax2 = plt.subplot(3, 3, 2)

scales = np.logspace(-6, 6, 50)
precision_losses = []
successful_scales = []

for scale in scales:
    try:
        test_val = torch.tensor([scale], dtype=torch.float32, device=device)
        quantized, state = bnb.quantize_fp4(test_val)
        dequantized = bnb.dequantize_fp4(quantized, state)
        
        if dequantized.item() != 0 and not np.isinf(dequantized.item()):
            rel_error = torch.abs(test_val - dequantized) / (torch.abs(test_val) + 1e-10)
            precision_losses.append(rel_error.item())
            successful_scales.append(scale)
    except:
        continue

ax2.loglog(successful_scales, precision_losses, 'o-', markersize=4, alpha=0.7)
ax2.set_xlabel('Value Scale')
ax2.set_ylabel('Relative Error')
ax2.set_title('Precision Loss vs Scale')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='1% Error')
ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='10% Error')
ax2.legend()

# ============================================================================
# Chart 3: Neural Network Weight Distribution Analysis
# ============================================================================
ax3 = plt.subplot(3, 3, 3)

# Generate typical neural network weights
weights = torch.randn(5000, dtype=torch.float32, device=device) * 0.1
quantized, state = bnb.quantize_fp4(weights)
dequantized = bnb.dequantize_fp4(quantized, state)

weights_np = weights.cpu().numpy()
dequantized_np = dequantized.cpu().numpy()

ax3.hist(weights_np, bins=100, alpha=0.6, label='Original Float32', density=True, color='blue')
ax3.hist(dequantized_np, bins=100, alpha=0.6, label='FP4 Quantized', density=True, color='red')
ax3.set_xlabel('Weight Value')
ax3.set_ylabel('Density')
ax3.set_title('Neural Network Weight Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add statistics text
mse = np.mean((weights_np - dequantized_np) ** 2)
zeros_created = np.sum(dequantized_np == 0) - np.sum(weights_np == 0)
ax3.text(0.02, 0.98, f'MSE: {mse:.2e}\nZeros created: {zeros_created}', 
         transform=ax3.transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

# ============================================================================
# Chart 4: Error Distribution
# ============================================================================
ax4 = plt.subplot(3, 3, 4)

errors = np.abs(weights_np - dequantized_np)
ax4.hist(errors, bins=100, alpha=0.7, color='red', edgecolor='darkred')
ax4.set_xlabel('Absolute Error')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Quantization Errors')
ax4.grid(True, alpha=0.3)
ax4.axvline(x=np.mean(errors), color='yellow', linestyle='--', label=f'Mean: {np.mean(errors):.2e}')
ax4.axvline(x=np.median(errors), color='orange', linestyle='--', label=f'Median: {np.median(errors):.2e}')
ax4.legend()

# ============================================================================
# Chart 5: Representable vs Non-Representable Values
# ============================================================================
ax5 = plt.subplot(3, 3, 5)

# Test a wide range of values
test_exponents = np.linspace(-10, 10, 1000)
test_vals = torch.tensor([10.0**exp for exp in test_exponents], dtype=torch.float32, device=device)

representable = []
non_representable = []
rep_exponents = []
non_rep_exponents = []

for i, val in enumerate(test_vals):
    try:
        quantized, state = bnb.quantize_fp4(val.unsqueeze(0))
        dequantized = bnb.dequantize_fp4(quantized, state)
        
        if dequantized.item() != 0 and not np.isinf(dequantized.item()):
            error = abs(val.item() - dequantized.item()) / (abs(val.item()) + 1e-10)
            if error < 0.5:  # Less than 50% error
                representable.append(val.item())
                rep_exponents.append(test_exponents[i])
            else:
                non_representable.append(val.item())
                non_rep_exponents.append(test_exponents[i])
        else:
            non_representable.append(val.item())
            non_rep_exponents.append(test_exponents[i])
    except:
        non_representable.append(val.item())
        non_rep_exponents.append(test_exponents[i])

ax5.scatter(rep_exponents, representable, c='green', alpha=0.6, s=1, label='Well Represented')
ax5.scatter(non_rep_exponents, non_representable, c='red', alpha=0.6, s=1, label='Poorly Represented')
ax5.set_xlabel('Exponent (10^x)')
ax5.set_ylabel('Value')
ax5.set_yscale('log')
ax5.set_title('FP4 Representability Map')
ax5.legend()
ax5.grid(True, alpha=0.3)

# ============================================================================
# Chart 6: Different Weight Initialization Schemes
# ============================================================================
ax6 = plt.subplot(3, 3, 6)

schemes = {
    'Xavier': torch.randn(1000, dtype=torch.float32, device=device) * 0.1,
    'He': torch.randn(1000, dtype=torch.float32, device=device) * 0.2,
    'Small': torch.randn(1000, dtype=torch.float32, device=device) * 0.01,
    'Large': torch.randn(1000, dtype=torch.float32, device=device) * 1.0,
}

scheme_names = []
mse_values = []
zero_counts = []

for name, weights in schemes.items():
    quantized, state = bnb.quantize_fp4(weights)
    dequantized = bnb.dequantize_fp4(quantized, state)
    
    mse = torch.mean((weights - dequantized) ** 2).item()
    zeros = (dequantized == 0).sum().item() - (weights == 0).sum().item()
    
    scheme_names.append(name)
    mse_values.append(mse)
    zero_counts.append(zeros)

x_pos = np.arange(len(scheme_names))
ax6_twin = ax6.twinx()

bars1 = ax6.bar(x_pos - 0.2, mse_values, 0.4, label='MSE', alpha=0.7, color='blue')
bars2 = ax6_twin.bar(x_pos + 0.2, zero_counts, 0.4, label='Zeros Created', alpha=0.7, color='red')

ax6.set_xlabel('Initialization Scheme')
ax6.set_ylabel('Mean Squared Error', color='blue')
ax6_twin.set_ylabel('Zeros Created', color='red')
ax6.set_title('Impact on Different Weight Schemes')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(scheme_names)
ax6.grid(True, alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars1, mse_values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.01,
             f'{val:.2e}', ha='center', va='bottom', fontsize=8)

for bar, val in zip(bars2, zero_counts):
    ax6_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(zero_counts)*0.01,
                  f'{val}', ha='center', va='bottom', fontsize=8)

# ============================================================================
# Chart 7: Correlation Plot - Original vs FP4
# ============================================================================
ax7 = plt.subplot(3, 3, 7)

# Use a subset for cleaner visualization
sample_weights = torch.randn(1000, dtype=torch.float32, device=device) * 0.1
quantized, state = bnb.quantize_fp4(sample_weights)
dequantized = bnb.dequantize_fp4(quantized, state)

orig_np = sample_weights.cpu().numpy()
deq_np = dequantized.cpu().numpy()

ax7.scatter(orig_np, deq_np, alpha=0.6, s=10)
min_val = min(orig_np.min(), deq_np.min())
max_val = max(orig_np.max(), deq_np.max())
ax7.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Match')

# Calculate and display correlation
correlation = np.corrcoef(orig_np, deq_np)[0, 1]
ax7.set_xlabel('Original Float32 Value')
ax7.set_ylabel('FP4 Quantized Value')
ax7.set_title(f'Original vs FP4 Correlation (r={correlation:.4f})')
ax7.legend()
ax7.grid(True, alpha=0.3)

# ============================================================================
# Chart 8: Memory Usage Comparison
# ============================================================================
ax8 = plt.subplot(3, 3, 8)

sizes = [1000, 10000, 100000, 1000000]
fp32_memory = [size * 4 for size in sizes]  # 4 bytes per float32
fp4_memory = [size * 0.5 for size in sizes]  # 0.5 bytes per fp4

x_pos = np.arange(len(sizes))
width = 0.35

bars1 = ax8.bar(x_pos - width/2, fp32_memory, width, label='Float32', alpha=0.7, color='blue')
bars2 = ax8.bar(x_pos + width/2, fp4_memory, width, label='FP4', alpha=0.7, color='green')

ax8.set_xlabel('Number of Parameters')
ax8.set_ylabel('Memory Usage (Bytes)')
ax8.set_title('Memory Usage: Float32 vs FP4')
ax8.set_xticks(x_pos)
ax8.set_xticklabels([f'{size:,}' for size in sizes])
ax8.set_yscale('log')
ax8.legend()
ax8.grid(True, alpha=0.3)

# Add savings percentage
for i, (fp32, fp4) in enumerate(zip(fp32_memory, fp4_memory)):
    savings = (1 - fp4/fp32) * 100
    ax8.text(i, max(fp32, fp4) * 1.5, f'{savings:.0f}% saved', 
             ha='center', va='bottom', fontweight='bold')

# ============================================================================
# Chart 9: Range Boundaries Visualization
# ============================================================================
ax9 = plt.subplot(3, 3, 9)

# Find the actual boundaries
boundary_exponents = np.arange(-15, 15, 0.1)
boundary_values = [10.0**exp for exp in boundary_exponents]

representable_boundary = []
underflow_boundary = []
overflow_boundary = []
rep_exp = []
under_exp = []
over_exp = []

for i, val in enumerate(boundary_values):
    try:
        test_tensor = torch.tensor([val], dtype=torch.float32, device=device)
        quantized, state = bnb.quantize_fp4(test_tensor)
        dequantized = bnb.dequantize_fp4(quantized, state)
        
        if dequantized.item() == 0:
            underflow_boundary.append(val)
            under_exp.append(boundary_exponents[i])
        elif np.isinf(dequantized.item()):
            overflow_boundary.append(val)
            over_exp.append(boundary_exponents[i])
        else:
            representable_boundary.append(val)
            rep_exp.append(boundary_exponents[i])
    except:
        continue

# Create boundary visualization
if underflow_boundary:
    ax9.fill_between(under_exp, [1e-20]*len(under_exp), underflow_boundary, 
                     alpha=0.3, color='red', label='Underflow (→0)')

if representable_boundary:
    ax9.fill_between(rep_exp, representable_boundary, [1e20]*len(rep_exp), 
                     alpha=0.3, color='green', label='Representable')

if overflow_boundary:
    ax9.fill_between(over_exp, overflow_boundary, [1e20]*len(over_exp), 
                     alpha=0.3, color='orange', label='Overflow (→∞)')

ax9.set_xlabel('Exponent (10^x)')
ax9.set_ylabel('Value')
ax9.set_yscale('log')
ax9.set_title('FP4 Representable Range Boundaries')
ax9.legend()
ax9.grid(True, alpha=0.3)

# Add boundary lines if we found them
if representable_boundary:
    min_rep = min(representable_boundary)
    max_rep = max(representable_boundary)
    ax9.axhline(y=min_rep, color='green', linestyle='--', alpha=0.8, 
                label=f'Min: {min_rep:.2e}')
    ax9.axhline(y=max_rep, color='green', linestyle='--', alpha=0.8, 
                label=f'Max: {max_rep:.2e}')

plt.tight_layout()
plt.savefig('fp4_comprehensive_analysis.png', dpi=300, bbox_inches='tight', 
            facecolor='black', edgecolor='none')
plt.show()

print("Analysis complete!")
print("=" * 60)
print("KEY VISUAL INSIGHTS:")
print("1. Chart 1: Shows exact range where FP4 fails (red/orange regions)")
print("2. Chart 2: Precision loss increases dramatically at extreme scales")
print("3. Chart 3: Neural network weights mostly survive quantization")
print("4. Chart 4: Most errors are small, but some are catastrophic")
print("5. Chart 5: Clear boundaries between representable/non-representable")
print("6. Chart 6: Different initialization schemes have varying impacts")
print("7. Chart 7: Strong correlation maintained in typical ranges")
print("8. Chart 8: 87.5% memory savings with FP4")
print("9. Chart 9: Visual map of FP4's representable range boundaries")
print("\nVisualization saved as 'fp4_comprehensive_analysis.png'")