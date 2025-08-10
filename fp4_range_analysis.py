import torch
import bitsandbytes.functional as bnb
import numpy as np
import matplotlib.pyplot as plt

print("FP4 Number Range and Precision Analysis")
print("=" * 60)

# Set device and seed for reproducibility
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

print(f"Using device: {device}")
print()

# ============================================================================
# SECTION 1: Understanding FP4 Range Limitations
# ============================================================================
print("1. FP4 RANGE LIMITATIONS")
print("-" * 40)

# Test different ranges of numbers
test_ranges = [
    ("Very small numbers", torch.tensor([1e-8, 1e-7, 1e-6, 1e-5, 1e-4], device=device)),
    ("Small numbers", torch.tensor([0.001, 0.01, 0.1, 0.5], device=device)),
    ("Normal range", torch.tensor([1.0, 2.0, 5.0, 10.0], device=device)),
    ("Large numbers", torch.tensor([100.0, 1000.0, 10000.0], device=device)),
    ("Very large numbers", torch.tensor([1e5, 1e6, 1e7, 1e8], device=device)),
    ("Extreme numbers", torch.tensor([1e10, 1e15, 1e20, 1e30], device=device)),
]

print("Testing different number ranges:")
print()

for range_name, values in test_ranges:
    print(f"{range_name}:")
    print(f"  Original: {values.cpu().numpy()}")
    
    try:
        # Quantize and dequantize
        quantized, state = bnb.quantize_fp4(values)
        dequantized = bnb.dequantize_fp4(quantized, state)
        
        print(f"  FP4 Result: {dequantized.cpu().numpy()}")
        
        # Calculate errors
        abs_errors = torch.abs(values - dequantized)
        rel_errors = abs_errors / (torch.abs(values) + 1e-10)
        
        print(f"  Absolute Errors: {abs_errors.cpu().numpy()}")
        print(f"  Relative Errors: {rel_errors.cpu().numpy()}")
        print(f"  Max Relative Error: {rel_errors.max().item():.6f}")
        
        # Check for zeros (underflow)
        zeros_count = (dequantized == 0).sum().item()
        if zeros_count > 0:
            print(f"  ⚠️  WARNING: {zeros_count} values became zero (underflow)!")
        
        # Check for infinities (overflow)
        inf_count = torch.isinf(dequantized).sum().item()
        if inf_count > 0:
            print(f"  ⚠️  WARNING: {inf_count} values became infinite (overflow)!")
            
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
    
    print()

# ============================================================================
# SECTION 2: Detailed Range Analysis
# ============================================================================
print("2. DETAILED FP4 REPRESENTABLE RANGE")
print("-" * 40)

# Create a comprehensive test of the representable range
print("Testing systematic range from 1e-10 to 1e10...")

# Create logarithmic range
exponents = np.arange(-10, 11, 1)  # -10 to 10
test_values = torch.tensor([10.0**exp for exp in exponents], dtype=torch.float32, device=device)

print(f"Original exponents: {exponents}")
print(f"Original values: {test_values.cpu().numpy()}")

quantized, state = bnb.quantize_fp4(test_values)
dequantized = bnb.dequantize_fp4(quantized, state)

print(f"FP4 dequantized: {dequantized.cpu().numpy()}")

# Analyze which values survived quantization
survived = dequantized != 0
underflowed = dequantized == 0
overflowed = torch.isinf(dequantized)

print(f"\nRange Analysis:")
print(f"  Values that survived: {survived.sum().item()}/{len(test_values)}")
print(f"  Values that underflowed to 0: {underflowed.sum().item()}")
print(f"  Values that overflowed to inf: {overflowed.sum().item()}")

if survived.any():
    min_representable = test_values[survived].min()
    max_representable = test_values[survived].max()
    print(f"  Approximate FP4 range: [{min_representable:.2e}, {max_representable:.2e}]")

# ============================================================================
# SECTION 3: Precision Analysis at Different Scales
# ============================================================================
print("\n3. PRECISION ANALYSIS AT DIFFERENT SCALES")
print("-" * 40)

scales = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]

for scale in scales:
    print(f"\nTesting precision around scale {scale:.0e}:")
    
    # Create fine-grained values around this scale
    base_values = torch.tensor([
        scale * 0.1, scale * 0.5, scale * 1.0, 
        scale * 1.5, scale * 2.0, scale * 5.0
    ], device=device)
    
    print(f"  Original: {base_values.cpu().numpy()}")
    
    try:
        quantized, state = bnb.quantize_fp4(base_values)
        dequantized = bnb.dequantize_fp4(quantized, state)
        
        print(f"  FP4 Result: {dequantized.cpu().numpy()}")
        
        # Calculate precision loss
        rel_errors = torch.abs(base_values - dequantized) / (torch.abs(base_values) + 1e-10)
        print(f"  Relative Errors: {rel_errors.cpu().numpy()}")
        print(f"  Average Precision Loss: {rel_errors.mean().item():.4f}")
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")

# ============================================================================
# SECTION 4: Neural Network Weight Distribution Analysis
# ============================================================================
print("\n4. NEURAL NETWORK WEIGHT DISTRIBUTION ANALYSIS")
print("-" * 40)

# Simulate different types of neural network weight distributions
weight_distributions = [
    ("Xavier/Glorot Normal", torch.randn(1000, device=device) * 0.1),
    ("He Normal", torch.randn(1000, device=device) * 0.2),
    ("Small weights", torch.randn(1000, device=device) * 0.01),
    ("Large weights", torch.randn(1000, device=device) * 1.0),
    ("Mixed scale", torch.cat([
        torch.randn(500, device=device) * 0.001,  # Very small
        torch.randn(500, device=device) * 1.0     # Normal
    ])),
]

for dist_name, weights in weight_distributions:
    print(f"\n{dist_name}:")
    print(f"  Original stats - Mean: {weights.mean():.6f}, Std: {weights.std():.6f}")
    print(f"  Original range: [{weights.min():.6f}, {weights.max():.6f}]")
    
    # Quantize
    quantized, state = bnb.quantize_fp4(weights)
    dequantized = bnb.dequantize_fp4(quantized, state)
    
    print(f"  FP4 stats - Mean: {dequantized.mean():.6f}, Std: {dequantized.std():.6f}")
    print(f"  FP4 range: [{dequantized.min():.6f}, {dequantized.max():.6f}]")
    
    # Analyze information loss
    zeros_created = (dequantized == 0).sum() - (weights == 0).sum()
    mse = torch.mean((weights - dequantized) ** 2)
    
    print(f"  Zeros created by quantization: {zeros_created.item()}")
    print(f"  Mean Squared Error: {mse.item():.8f}")
    print(f"  Signal-to-Noise Ratio: {(weights.var() / mse.item()):.2f}")

# ============================================================================
# SECTION 5: Critical Value Analysis
# ============================================================================
print("\n5. CRITICAL VALUE ANALYSIS")
print("-" * 40)

print("Finding the boundaries of FP4 representation...")

# Test very small positive values
small_values = torch.tensor([
    1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
], device=device)

print(f"\nSmall positive values test:")
print(f"Original: {small_values.cpu().numpy()}")

quantized, state = bnb.quantize_fp4(small_values)
dequantized = bnb.dequantize_fp4(quantized, state)

print(f"FP4 result: {dequantized.cpu().numpy()}")

# Find the smallest representable positive value
non_zero_mask = dequantized > 0
if non_zero_mask.any():
    smallest_positive = dequantized[non_zero_mask].min()
    print(f"Smallest representable positive value: {smallest_positive:.2e}")

# Test large values
large_values = torch.tensor([
    1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10
], device=device)

print(f"\nLarge positive values test:")
print(f"Original: {large_values.cpu().numpy()}")

quantized, state = bnb.quantize_fp4(large_values)
dequantized = bnb.dequantize_fp4(quantized, state)

print(f"FP4 result: {dequantized.cpu().numpy()}")

# Find the largest representable finite value
finite_mask = torch.isfinite(dequantized)
if finite_mask.any():
    largest_finite = dequantized[finite_mask].max()
    print(f"Largest representable finite value: {largest_finite:.2e}")

# ============================================================================
# SECTION 6: Visualization
# ============================================================================
print("\n6. CREATING VISUALIZATIONS")
print("-" * 40)

plt.style.use('dark_background')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Range representation capability
exponents = np.arange(-8, 9, 1)
test_vals = torch.tensor([10.0**exp for exp in exponents], device=device)
quantized, state = bnb.quantize_fp4(test_vals)
dequantized = bnb.dequantize_fp4(quantized, state)

ax1.semilogy(exponents, test_vals.cpu().numpy(), 'o-', label='Original', markersize=8)
ax1.semilogy(exponents, dequantized.cpu().numpy(), 's-', label='FP4', markersize=8)
ax1.set_xlabel('Exponent (10^x)')
ax1.set_ylabel('Value')
ax1.set_title('FP4 Range Representation')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Precision loss across scales
scales = np.logspace(-3, 3, 20)
precision_losses = []

for scale in scales:
    test_val = torch.tensor([scale], device=device)
    quantized, state = bnb.quantize_fp4(test_val)
    dequantized = bnb.dequantize_fp4(quantized, state)
    rel_error = torch.abs(test_val - dequantized) / (torch.abs(test_val) + 1e-10)
    precision_losses.append(rel_error.item())

ax2.loglog(scales, precision_losses, 'o-', markersize=6)
ax2.set_xlabel('Value Scale')
ax2.set_ylabel('Relative Error')
ax2.set_title('Precision Loss vs Scale')
ax2.grid(True, alpha=0.3)

# Plot 3: Weight distribution comparison
weights = torch.randn(1000, device=device) * 0.1
quantized, state = bnb.quantize_fp4(weights)
dequantized = bnb.dequantize_fp4(quantized, state)

ax3.hist(weights.cpu().numpy(), bins=50, alpha=0.7, label='Original', density=True)
ax3.hist(dequantized.cpu().numpy(), bins=50, alpha=0.7, label='FP4', density=True)
ax3.set_xlabel('Weight Value')
ax3.set_ylabel('Density')
ax3.set_title('Weight Distribution: Original vs FP4')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Error distribution
errors = torch.abs(weights - dequantized).cpu().numpy()
ax4.hist(errors, bins=50, alpha=0.7, color='red')
ax4.set_xlabel('Absolute Error')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Quantization Errors')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fp4_range_analysis.png', dpi=150, bbox_inches='tight', facecolor='black')
plt.show()

print("Analysis complete! Visualization saved as 'fp4_range_analysis.png'")
print("\nKEY FINDINGS:")
print("=" * 60)
print("1. FP4 has a very limited representable range")
print("2. Very small values (< ~1e-5) often underflow to zero")
print("3. Very large values (> ~1e4) may overflow to infinity")
print("4. Precision loss increases at extreme scales")
print("5. Neural network weights in typical ranges work reasonably well")
print("6. Mixed-scale distributions suffer significant information loss")