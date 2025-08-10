import torch
import torch.nn as nn
# Import the bitsandbytes library, specifically its neural network modules
import bitsandbytes.optim as bnb_optim
import bitsandbytes.nn as bnb_nn

# Ensure CUDA is available, as bitsandbytes is CUDA-only
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for bitsandbytes 4-bit quantization.")

# Check for BFloat16 support, which is crucial for 4-bit training performance
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("Your GPU does not support BFloat16, which is highly recommended for 4-bit training.")

device = 'cuda'
print(f"Using device: {device}")

# --- The "Real" 4-Bit Neural Network with FP4 ---
class NetWith4BitLinear(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=1):
        super().__init__()
        
        # Method 1: Using LinearFP4 directly (simpler syntax)
        self.fc1 = bnb_nn.LinearFP4(
            input_size,
            hidden_size,
            bias=True,
            compute_dtype=torch.bfloat16,  # Perform matmul in BFloat16
            compress_statistics=True,  # Compress statistics for memory efficiency
            quant_storage=torch.uint8  # Storage type for quantized weights
        )
        
        # Method 2: Using Linear4bit with quant_type='fp4' (equivalent to LinearFP4)
        self.fc2 = bnb_nn.Linear4bit(
            hidden_size,
            hidden_size,
            bias=True,
            quant_type='fp4',  # Explicitly specify FP4 quantization
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
            quant_storage=torch.uint8
        )
        
        # The final layer can be a standard linear layer or also a 4-bit one.
        # Using FP4 for consistency
        self.fc3 = bnb_nn.LinearFP4(
            hidden_size,
            output_size,
            bias=True,
            compute_dtype=torch.bfloat16,
            compress_statistics=True
        )
    
    def forward(self, x):
        # The input should be cast to the compute dtype (BF16)
        x = x.to(torch.bfloat16)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_simple_sum_data(batch_size, device='cuda'):
    """Generate data for learning simple addition: (a, b) -> a + b"""
    a = torch.randint(1, 10, (batch_size, 1), dtype=torch.float32, device=device)
    b = torch.randint(1, 10, (batch_size, 1), dtype=torch.float32, device=device)
    x = torch.cat([a, b], dim=1)
    y = a + b
    return x, y

# --- Create and Initialize Model ---
# First create the model with FP16/BF16 weights
model = NetWith4BitLinear().to(device)

# Optional: If you want to load pre-trained FP16/BF16 weights, do it before quantization
# model.load_state_dict(pretrained_weights)

# The quantization happens when we move the model to CUDA
# This is when the FP16/BF16 weights get quantized to FP4
print("Model created with FP4 quantization")

# You can inspect a layer to see how it's different from a standard nn.Linear
print("\nInspecting the FP4 linear layers:")
print(f"FC1 (LinearFP4): {model.fc1}")
print(f"FC2 (Linear4bit with fp4): {model.fc2}")
print(f"FC3 (LinearFP4): {model.fc3}")
print("\n")

# Use a bitsandbytes optimizer (like AdamW8bit) which is optimized for this kind of training.
# It correctly handles the FP32 master weights.
optimizer = bnb_optim.AdamW8bit(model.parameters(), lr=0.01, betas=(0.9, 0.995))
criterion = nn.MSELoss()

print("Training with FP4 weight storage and BF16 computation...")
model.train()

for epoch in range(2001):
    x, y = generate_simple_sum_data(256, device=device)
    
    # Forward pass
    output = model(x)
    
    # Loss must be calculated in FP32 for stability
    loss = criterion(output.float(), y.float())
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# --- Final Evaluation ---
print("\n" + "="*50)
print("Testing Final Model with FP4 4-Bit Layers")
print("="*50)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for a_val in range(1, 10):
        for b_val in range(1, 10):
            x_test = torch.tensor([[a_val, b_val]], device=device, dtype=torch.float32)
            # The model internally handles the conversion to BF16
            pred = model(x_test).item()
            pred_rounded = round(pred)
            actual = a_val + b_val
            
            if pred_rounded == actual:
                correct += 1
            total += 1
            
            if a_val <= 4 and b_val <= 4:
                status = "✓" if pred_rounded == actual else "✗"
                print(f"{a_val} + {b_val} = {actual},  Pred: {pred:.2f} -> {pred_rounded}  {status}")

    accuracy = 100 * correct / total
    print(f"\nFinal Accuracy with FP4 quantization (rounded prediction): {accuracy:.2f}%")

# Optional: Print memory usage comparison
print("\n" + "="*50)
print("Memory Efficiency Information")
print("="*50)
print("FP4 uses 4 bits per weight vs 32 bits for FP32 (8x reduction)")
print("FP4 uses 4 bits per weight vs 16 bits for FP16 (4x reduction)")
print("With compress_statistics=True, additional memory savings are achieved")