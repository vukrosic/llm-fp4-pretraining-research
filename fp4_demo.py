import torch
import bitsandbytes.functional as bnb
import numpy as np

print("FP4 Quantization Step-by-Step Demo")
print("=" * 40)

# Create a simple range of float32 values
original_values = torch.tensor([
    -2.5, -1.8, -0.9, -0.1, 0.0, 0.1, 0.9, 1.8, 2.5, 3.7
], dtype=torch.float32, device='cuda')

print("1. Original float32 values:")
print(f"   {original_values.cpu().numpy()}")
print(f"   Memory: {original_values.numel() * 4} bytes (32 bits each)")

# Step 2: Quantize to FP4
print("\n2. Quantizing to FP4...")
quantized, state = bnb.quantize_fp4(original_values)

print(f"   Quantized tensor shape: {quantized.shape}")
print(f"   Quantized memory: {quantized.numel()} bytes (4 bits each)")
print(f"   Memory reduction: {(1 - quantized.numel() / (original_values.numel() * 4)) * 100:.1f}%")
print(f"   State info: {type(state)}")

# Step 3: Dequantize back to float32
print("\n3. Dequantizing back to float32...")
dequantized = bnb.dequantize_fp4(quantized, state)

print(f"   Dequantized values:")
print(f"   {dequantized.cpu().numpy()}")

# Step 4: Compare precision loss
print("\n4. Precision Analysis:")
print("   Original  ->  Dequantized  ->  Error")
for i, (orig, deq) in enumerate(zip(original_values.cpu(), dequantized.cpu())):
    error = abs(orig - deq)
    print(f"   {orig:8.3f}  ->  {deq:8.3f}     ->  {error:.6f}")

# Step 5: Overall statistics
mse = torch.mean((original_values - dequantized) ** 2)
max_error = torch.max(torch.abs(original_values - dequantized))

print(f"\n5. Summary:")
print(f"   Mean Squared Error: {mse:.8f}")
print(f"   Maximum Error: {max_error:.6f}")
print(f"   Memory saved: ~75% (32-bit -> 4-bit)")