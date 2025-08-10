import torch
import torch.nn as nn
import bitsandbytes.functional as bnb

# Mixed Precision FP4 Neural Network
class FP4Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Initialize weights in BF16 for better precision during training
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        
        # Store FP4 quantized version for inference
        self.quantized_weight = None
        self.quant_state = None
        
    def quantize_weights(self):
        """Quantize weights to FP4 for storage/inference"""
        if not self.weight.is_cuda:
            raise RuntimeError("Weights must be on CUDA device for FP4 quantization")
        # Convert to FP32 for quantization (bitsandbytes requirement)
        self.quantized_weight, self.quant_state = bnb.quantize_fp4(self.weight.data.float())
    
    def forward(self, x):
        # Always use full precision weights during training
        # Only use quantized weights during inference if explicitly set
        if self.training:
            # Use BF16 weights during training
            return torch.matmul(x, self.weight.t()) + self.bias
        else:
            # During inference, can optionally use quantized weights
            if self.quantized_weight is not None:
                weight = bnb.dequantize_fp4(self.quantized_weight, self.quant_state).to(x.dtype)
            else:
                weight = self.weight
            return torch.matmul(x, weight.t()) + self.bias

class MixedPrecisionNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=1):
        super().__init__()
        # Use larger hidden size for better capacity with quantization
        self.fc1 = FP4Linear(input_size, hidden_size)
        self.fc2 = FP4Linear(hidden_size, hidden_size)
        self.fc3 = FP4Linear(hidden_size, output_size)
        
        # Layer normalization to stabilize training
        self.ln1 = nn.LayerNorm(hidden_size, dtype=torch.bfloat16)
        self.ln2 = nn.LayerNorm(hidden_size, dtype=torch.bfloat16)
    
    def forward(self, x):
        x = x.to(torch.bfloat16)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)  # No sigmoid, use raw output
        return x
    
    def quantize_all_weights(self):
        """Quantize all layer weights to FP4 for inference"""
        self.fc1.quantize_weights()
        self.fc2.quantize_weights()
        self.fc3.quantize_weights()

def generate_simple_sum_data(batch_size, device='cuda'):
    """Generate data for learning simple addition: (a, b) -> a + b where a,b in [1,5]"""
    a = torch.randint(1, 6, (batch_size,), dtype=torch.float32, device=device)
    b = torch.randint(1, 6, (batch_size,), dtype=torch.float32, device=device)
    
    # Keep inputs as raw values (1-5), don't normalize
    x = torch.stack([a, b], dim=1)
    # Target is the actual sum (2-10)
    y = a + b
    
    return x, y

# Create and test network
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for FP4 quantization")

device = 'cuda'
print(f"Using device: {device}")

# Create model with mixed precision
model = MixedPrecisionNet(input_size=2, hidden_size=64, output_size=1).to(device)

# Test forward pass
test_x, test_y = generate_simple_sum_data(32, device)
output = model(test_x)
print(f"Output shape: {output.shape}")
print(f"Sample input: {test_x[0].cpu().numpy()} -> sum: {test_y[0].item()}")
print(f"Model output (before training): {output[0].item():.2f}")

# Training with BF16 weights
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

print("\nTraining with BF16 weights...")
model.train()
best_accuracy = 0

for epoch in range(1000):
    # Generate training data
    x, y = generate_simple_sum_data(256, device)
    
    # Forward pass
    output = model(x).squeeze()
    loss = criterion(output, y.to(torch.bfloat16))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()
    
    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            test_x, test_y = generate_simple_sum_data(500, device)
            test_pred = model(test_x).squeeze()
            # Round predictions to nearest integer
            test_pred_rounded = torch.round(test_pred)
            accuracy = (test_pred_rounded == test_y.to(torch.bfloat16)).float().mean()
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
        
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Accuracy: {accuracy:.3f}, Best: {best_accuracy:.3f}")
        model.train()

print(f"\nBest training accuracy with BF16: {best_accuracy:.3f}")

# Now quantize to FP4 and test
print("\nQuantizing weights to FP4 for inference...")
model.eval()
model.quantize_all_weights()

# Test both BF16 and FP4 inference
print("\nTesting on all possible sums:")
print("=" * 60)

with torch.no_grad():
    # Test with BF16 weights (before quantization effect)
    model.fc1.quantized_weight = None
    model.fc2.quantized_weight = None
    model.fc3.quantized_weight = None
    
    bf16_correct = 0
    total_tests = 0
    
    print("BF16 Inference Results:")
    print("-" * 30)
    for a in range(1, 6):
        for b in range(1, 6):
            x = torch.tensor([[a, b]], device=device, dtype=torch.float32)
            pred = model(x).item()
            pred_rounded = round(pred)
            actual = a + b
            error = abs(pred - actual)
            is_correct = pred_rounded == actual
            bf16_correct += is_correct
            total_tests += 1
            
            if a <= 3 and b <= 3:  # Show subset of results
                status = "✓" if is_correct else "✗"
                print(f"{a} + {b} = {actual}, Pred: {pred:.2f} → {pred_rounded}, Error: {error:.2f} {status}")
    
    print(f"\nBF16 Accuracy: {bf16_correct}/{total_tests} = {bf16_correct/total_tests:.1%}")
    
    # Now test with FP4 quantized weights
    model.quantize_all_weights()
    
    fp4_correct = 0
    total_tests = 0
    
    print("\nFP4 Inference Results:")
    print("-" * 30)
    for a in range(1, 6):
        for b in range(1, 6):
            x = torch.tensor([[a, b]], device=device, dtype=torch.float32)
            pred = model(x).item()
            pred_rounded = round(pred)
            actual = a + b
            error = abs(pred - actual)
            is_correct = pred_rounded == actual
            fp4_correct += is_correct
            total_tests += 1
            
            if a <= 3 and b <= 3:  # Show subset of results
                status = "✓" if is_correct else "✗"
                print(f"{a} + {b} = {actual}, Pred: {pred:.2f} → {pred_rounded}, Error: {error:.2f} {status}")
    
    print(f"\nFP4 Accuracy: {fp4_correct}/{total_tests} = {fp4_correct/total_tests:.1%}")
    print(f"Accuracy drop from quantization: {(bf16_correct - fp4_correct)/total_tests:.1%}")

# Memory comparison
print("\n" + "=" * 60)
print("Memory Usage Comparison:")
print("-" * 30)

# Calculate parameter sizes
total_params = sum(p.numel() for p in model.parameters())
bf16_size = total_params * 2  # 2 bytes per BF16 parameter
fp4_size = total_params * 0.5  # 0.5 bytes per FP4 parameter (4 bits)

print(f"Total parameters: {total_params:,}")
print(f"BF16 model size: {bf16_size / 1024:.2f} KB")
print(f"FP4 model size: {fp4_size / 1024:.2f} KB")
print(f"Compression ratio: {bf16_size / fp4_size:.1f}x")