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
        # Dequantize and compute
        weight = bnb.dequantize_fp4(self.quantized_weight, self.quant_state)
        return torch.matmul(x, weight.t()) + self.bias

class FP4Net(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=1):
        super().__init__()
        self.fc1 = FP4Linear(input_size, hidden_size)
        self.fc2 = FP4Linear(hidden_size, hidden_size)
        self.fc3 = FP4Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_mod_division_data(batch_size, max_val=100):
    """Generate data for learning modular division: (a, b) -> a % b"""
    # Avoid division by zero by ensuring b is at least 1
    a = torch.randint(1, max_val, (batch_size,), dtype=torch.float32)
    b = torch.randint(1, max_val, (batch_size,), dtype=torch.float32)
    
    # Normalize inputs to help with training
    x = torch.stack([a / max_val, b / max_val], dim=1)
    y = (a % b) / max_val  # Normalize output too
    
    return x, y

# Create and test network
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for FP4 quantization. CPU only supports NF4.")

device = 'cuda'
print(f"Using device: {device}")
model = FP4Net(input_size=2, hidden_size=64, output_size=1).to(device)

# Quantize weights after moving to GPU
model.fc1.quantize_weights()
model.fc2.quantize_weights()
model.fc3.quantize_weights()

# Test forward pass
test_x, test_y = generate_mod_division_data(32)
test_x, test_y = test_x.to(device), test_y.to(device)
output = model(test_x)
print(f"Output shape: {output.shape}")
print(f"Sample input: {test_x[0].cpu().numpy()}")
print(f"Expected output: {test_y[0].item():.4f}")
print(f"Model output: {output[0].item():.4f}")

# Training loop for modular division
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\nTraining to learn modular division (a % b)...")
for epoch in range(100):
    # Generate training data
    x, y = generate_mod_division_data(128)
    x, y = x.to(device), y.to(device)
    
    # Forward
    output = model(x)
    loss = criterion(output.squeeze(), y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Re-quantize weights after update
    model.fc1.quantize_weights()
    model.fc2.quantize_weights()
    model.fc3.quantize_weights()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Test the trained model
print("\nTesting trained model:")
test_cases = [
    (17, 5),   # 17 % 5 = 2
    (23, 7),   # 23 % 7 = 2  
    (50, 13),  # 50 % 13 = 11
    (99, 10),  # 99 % 10 = 9
]

model.eval()
with torch.no_grad():
    for a, b in test_cases:
        # Normalize input
        x = torch.tensor([[a/100, b/100]], device=device)
        pred = model(x).item() * 100  # Denormalize output
        actual = a % b
        print(f"{a} % {b} = {actual}, Model prediction: {pred:.2f}, Error: {abs(pred - actual):.2f}")