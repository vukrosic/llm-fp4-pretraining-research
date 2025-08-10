import torch
import numpy as np

def simple_fp4_quantize(tensor):
    """
    Simple FP4 quantization from scratch
    FP4 format: 1 sign bit + 2 exponent bits + 1 mantissa bit
    """
    # Convert to numpy for easier bit manipulation
    values = tensor.cpu().numpy().flatten()
    quantized = np.zeros_like(values, dtype=np.uint8)
    
    for i, val in enumerate(values):
        if val == 0:
            quantized[i] = 0
            continue
            
        # Extract sign
        sign = 0 if val >= 0 else 1
        abs_val = abs(val)
        
        # Find exponent (bias = 1 for 2-bit exponent)
        if abs_val >= 2.0:
            exp = 3  # 11 in binary (max)
            mantissa = 1 if abs_val >= 4.0 else 0
        elif abs_val >= 1.0:
            exp = 2  # 10 in binary
            mantissa = 1 if abs_val >= 1.5 else 0
        elif abs_val >= 0.5:
            exp = 1  # 01 in binary
            mantissa = 1 if abs_val >= 0.75 else 0
        else:
            exp = 0  # 00 in binary
            mantissa = 1 if abs_val >= 0.25 else 0
            
        # Pack into 4 bits: sign(1) + exp(2) + mantissa(1)
        quantized[i] = (sign << 3) | (exp << 1) | mantissa
    
    return quantized

def simple_fp4_dequantize(quantized, original_shape):
    """Dequantize FP4 back to float32"""
    dequantized = np.zeros(len(quantized), dtype=np.float32)
    
    for i, q_val in enumerate(quantized):
        if q_val == 0:
            dequantized[i] = 0.0
            continue
            
        # Unpack bits
        sign = (q_val >> 3) & 1
        exp = (q_val >> 1) & 3
        mantissa = q_val & 1
        
        # Convert back to float
        if exp == 3:
            value = 2.0 + mantissa * 2.0  # [2.0, 4.0]
        elif exp == 2:
            value = 1.0 + mantissa * 0.5  # [1.0, 1.5]
        elif exp == 1:
            value = 0.5 + mantissa * 0.25  # [0.5, 0.75]
        else:  # exp == 0
            value = 0.125 + mantissa * 0.125  # [0.125, 0.25]
            
        dequantized[i] = -value if sign else value
    
    return torch.tensor(dequantized.reshape(original_shape))

# Test the implementation
if __name__ == "__main__":
    # Create test tensor
    original = torch.tensor([0.1, 0.5, 1.0, 2.0, -0.3, -1.5, 3.0])
    print(f"Original: {original}")
    
    # Quantize
    quantized = simple_fp4_quantize(original)
    print(f"Quantized (4-bit): {quantized}")
    
    # Dequantize
    reconstructed = simple_fp4_dequantize(quantized, original.shape)
    print(f"Reconstructed: {reconstructed}")
    
    # Calculate error
    error = torch.abs(original - reconstructed)
    print(f"Absolute error: {error}")
    print(f"Max error: {error.max().item():.4f}")
    
    # Memory savings
    original_bits = original.numel() * 32  # float32
    quantized_bits = len(quantized) * 4   # 4-bit
    savings = (1 - quantized_bits / original_bits) * 100
    print(f"Memory savings: {savings:.1f}%")