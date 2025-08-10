import torch
import bitsandbytes.functional as bnb
import numpy as np
import matplotlib.pyplot as plt

print("FP4 Quantization Step-by-Step Demo")
print("=" * 50)

# Create neural network-like weight distribution (normal distribution)
torch.manual_seed(42)
n_weights = 50
original_values = torch.randn(n_weights, dtype=torch.float32, device='cuda') * 0.5

print("1. Original float32 values (neural network weights):")
print(f"   Shape: {original_values.shape}")
print(f"   Range: [{original_values.min():.4f}, {original_values.max():.4f}]")
print(f"   Mean: {original_values.mean():.4f}, Std: {original_values.std():.4f}")
print(f"   Memory: {original_values.numel() * 4} bytes (32 bits each)")
print(f"   First 10 values: {original_values[:10].cpu().numpy()}")

# Step 2: Quantize to FP4
print("\n2. Quantizing to FP4...")
quantized, state = bnb.quantize_fp4(original_values)

print(f"   Quantized tensor shape: {quantized.shape}")
print(f"   Quantized memory: {quantized.numel()} bytes (4 bits each)")
print(f"   Memory reduction: {(1 - quantized.numel() / (original_values.numel() * 4)) * 100:.1f}%")

# Examine the state object
print(f"\n3. State Object Analysis:")
print(f"   State type: {type(state)}")
if hasattr(state, '__dict__'):
    for key, value in state.__dict__.items():
        if torch.is_tensor(value):
            print(f"   {key}: tensor shape {value.shape}, dtype {value.dtype}")
            if value.numel() <= 10:
                print(f"      values: {value.cpu().numpy()}")
        else:
            print(f"   {key}: {value}")

# Step 3: Dequantize back to float32
print("\n4. Dequantizing back to float32...")
dequantized = bnb.dequantize_fp4(quantized, state)

print(f"   Dequantized shape: {dequantized.shape}")
print(f"   Range: [{dequantized.min():.4f}, {dequantized.max():.4f}]")
print(f"   First 10 values: {dequantized[:10].cpu().numpy()}")

# Step 4: Analyze errors
errors = torch.abs(original_values - dequantized)
relative_errors = errors / (torch.abs(original_values) + 1e-8)

print(f"\n5. Error Analysis:")
print(f"   Mean Absolute Error: {errors.mean():.6f}")
print(f"   Max Absolute Error: {errors.max():.6f}")
print(f"   Mean Relative Error: {relative_errors.mean():.6f}")
print(f"   Max Relative Error: {relative_errors.max():.6f}")

# Create visualizations
plt.style.use('dark_background')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Original vs Dequantized values
orig_np = original_values.cpu().numpy()
deq_np = dequantized.cpu().numpy()
indices = np.arange(len(orig_np))

ax1.scatter(indices, orig_np, alpha=0.7, label='Original', s=30)
ax1.scatter(indices, deq_np, alpha=0.7, label='Dequantized', s=30)
ax1.set_title('Original vs Dequantized Values')
ax1.set_xlabel('Weight Index')
ax1.set_ylabel('Value')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Error distribution
error_np = errors.cpu().numpy()
ax2.bar(indices, error_np, alpha=0.7, color='red')
ax2.set_title('Absolute Errors')
ax2.set_xlabel('Weight Index')
ax2.set_ylabel('Error')
ax2.grid(True, alpha=0.3)

# Plot 3: Histogram comparison
ax3.hist(orig_np, bins=15, alpha=0.7, label='Original', density=True)
ax3.hist(deq_np, bins=15, alpha=0.7, label='Dequantized', density=True)
ax3.set_title('Value Distribution')
ax3.set_xlabel('Value')
ax3.set_ylabel('Density')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Scatter plot original vs dequantized
ax4.scatter(orig_np, deq_np, alpha=0.7)
min_val = min(orig_np.min(), deq_np.min())
max_val = max(orig_np.max(), deq_np.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect match')
ax4.set_title('Original vs Dequantized Correlation')
ax4.set_xlabel('Original Value')
ax4.set_ylabel('Dequantized Value')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fp4_quantization_analysis.png', dpi=150, bbox_inches='tight', facecolor='black')
plt.show()

print(f"\n6. Summary:")
print(f"   Memory saved: ~75% (32-bit -> 4-bit)")
print(f"   Precision trade-off: {errors.mean():.6f} average error")
print(f"   Visualization saved as 'fp4_quantization_analysis.png'")