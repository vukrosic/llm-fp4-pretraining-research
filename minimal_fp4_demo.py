#!/usr/bin/env python3
"""
Minimal FP4 Quantization Demo - Shows exactly what happens during quantization
"""
import torch
import bitsandbytes as bnb

# Create some example weights (like neural network parameters)
print("🔬 FP4 Quantization Demo")
print("=" * 40)

# Original BF16 weights
weights = torch.tensor([0.1234, -0.5678, 0.9012, -0.3456, 0.7890], dtype=torch.bfloat16).cuda()
print(f"Original BF16 weights: {weights}")
print(f"Memory usage: {weights.numel() * 2} bytes (2 bytes per BF16)")

# Quantize to FP4
quantized, state = bnb.quantize_fp4(weights)
print(f"\nQuantized FP4 data: {quantized}")
print(f"Memory usage: {quantized.numel() * 0.5} bytes (0.5 bytes per FP4)")
print(f"Quantization state: {state}")

# Dequantize back to BF16
dequantized = bnb.dequantize_fp4(quantized, state)
print(f"\nDequantized weights: {dequantized}")

# Show the error
error = torch.abs(weights.float() - dequantized.float())
print(f"Quantization error: {error}")
print(f"Max error: {error.max().item():.6f}")
print(f"Mean error: {error.mean().item():.6f}")

print(f"\n💾 Memory savings: {((weights.numel() * 2 - quantized.numel() * 0.5) / (weights.numel() * 2) * 100):.1f}%")