import torch
import torch.nn as nn
import bitsandbytes.functional as bnb

# Simple FP4 Neural Network using bitsandbytes
class FP4Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Initialize weights in FP32
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Quantize to FP4 - will be done after moving to GPU
        self.quantized_weight = None
        self.quant_state = None
    
    def quantize_weights(self):
        # Ensure weights are on GPU before quantization
        if not self.weight.is_cuda:
            raise RuntimeError("Weights must be on CUDA device for FP4 quantization")
        # Quantize weights to FP4 using bitsandbytes
        self.quantized_weight, self.quant_state = bnb.quantize_fp4(self.weight.data)
    
    def forward(self, x):
        # Use quantized weights if available, otherwise use FP32 weights
        if self.quantized_weight is not None:
            weight = bnb.dequantize_fp4(self.quantized_weight, self.quant_state)
        else:
            weight = self.weight
        return torch.matmul(x, weight.t()) + self.bias

class FP4Net(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=1):
        super().__init__()
        self.fc1 = FP4Linear(input_size, hidden_size)
        self.fc2 = FP4Linear(hidden_size, hidden_size)
        self.fc3 = FP4Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid to keep output in [0,1]
        return x

def generate_simple_sum_data(batch_size):
    """Generate data for learning simple addition: (a, b) -> a + b where a,b in [1,5]"""
    a = torch.randint(1, 6, (batch_size,), dtype=torch.float32)  # 1 to 5
    b = torch.randint(1, 6, (batch_size,), dtype=torch.float32)  # 1 to 5
    
    # Normalize inputs to [0, 1] (1->0.2, 5->1.0)
    x = torch.stack([(a - 1) / 4, (b - 1) / 4], dim=1)
    # Target is the sum, normalized (2->0.1, 10->1.0)
    y = (a + b - 2) / 8.0
    
    return x, y

# Create and test network
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for FP4 quantization. CPU only supports NF4.")

device = 'cuda'
print(f"Using device: {device}")
model = FP4Net(input_size=2, hidden_size=32, output_size=1).to(device)

# Quantize weights immediately for FP4-only training
model.fc1.quantize_weights()
model.fc2.quantize_weights()
model.fc3.quantize_weights()

# Test forward pass
test_x, test_y = generate_simple_sum_data(32)
test_x, test_y = test_x.to(device), test_y.to(device)
output = model(test_x)
print(f"Output shape: {output.shape}")
print(f"Sample input: {test_x[0].cpu().numpy()} -> numbers: {test_x[0].cpu().numpy() * 4 + 1}")
print(f"Expected output: {test_y[0].item():.4f} -> sum: {test_y[0].item() * 8 + 2:.1f}")
print(f"Model output: {output[0].item():.4f} -> predicted sum: {output[0].item() * 8 + 2:.1f}")

# Training loop for simple addition
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("\nTraining to learn simple addition (1-5 + 1-5) with FP4 weights...")
for epoch in range(500):
    # Generate training data
    x, y = generate_simple_sum_data(128)
    x, y = x.to(device), y.to(device)
    
    # Forward
    output = model(x)
    loss = criterion(output.squeeze(), y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Re-quantize weights every 10 epochs
    if epoch % 10 == 0:
        model.fc1.quantize_weights()
        model.fc2.quantize_weights()
        model.fc3.quantize_weights()
    
    if epoch % 100 == 0:
        # Calculate actual accuracy on denormalized values
        with torch.no_grad():
            test_x, test_y = generate_simple_sum_data(100)
            test_x, test_y = test_x.to(device), test_y.to(device)
            test_pred = model(test_x).squeeze() * 8 + 2  # Denormalize to [2,10]
            test_actual = test_y * 8 + 2
            accuracy = (torch.abs(test_pred - test_actual) < 0.5).float().mean()
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Accuracy: {accuracy:.3f}")

# Test the trained model on all possible combinations
print("\nTesting trained model on all possible sums (1-5 + 1-5):")
model.eval()
with torch.no_grad():
    all_correct = 0
    total_tests = 0
    
    for a in range(1, 6):
        for b in range(1, 6):
            # Normalize input
            x = torch.tensor([[(a-1)/4, (b-1)/4]], device=device)
            pred = model(x).item() * 8 + 2  # Denormalize output
            actual = a + b
            error = abs(pred - actual)
            is_correct = error < 0.5
            all_correct += is_correct
            total_tests += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{a} + {b} = {actual}, Prediction: {pred:.1f}, Error: {error:.1f} {status}")
    
    print(f"\nOverall Accuracy: {all_correct}/{total_tests} = {all_correct/total_tests:.1%}")