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
    def __init__(self, input_size=2, hidden_size=128, output_size=1):
        super().__init__()
        self.fc1 = FP4Linear(input_size, hidden_size)
        self.fc2 = FP4Linear(hidden_size, hidden_size)
        self.fc3 = FP4Linear(hidden_size, hidden_size)
        self.fc4 = FP4Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Sigmoid to keep output in [0,1]
        return x

def generate_arithmetic_data(batch_size, max_val=50):
    """Generate data for learning: (a, b) -> (a + b) % 10 (last digit of sum)"""
    a = torch.randint(0, max_val, (batch_size,), dtype=torch.float32)
    b = torch.randint(0, max_val, (batch_size,), dtype=torch.float32)
    
    # Normalize inputs to [0, 1]
    x = torch.stack([a / max_val, b / max_val], dim=1)
    # Target is the last digit of the sum, normalized to [0, 1]
    y = ((a + b) % 10) / 10.0
    
    return x, y

# Create and test network
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for FP4 quantization. CPU only supports NF4.")

device = 'cuda'
print(f"Using device: {device}")
model = FP4Net(input_size=2, hidden_size=128, output_size=1).to(device)

# Don't quantize initially - let it learn first

# Test forward pass
test_x, test_y = generate_arithmetic_data(32)
test_x, test_y = test_x.to(device), test_y.to(device)
output = model(test_x)
print(f"Output shape: {output.shape}")
print(f"Sample input: {test_x[0].cpu().numpy()}")
print(f"Expected output: {test_y[0].item():.4f}")
print(f"Model output: {output[0].item():.4f}")

# Training loop for arithmetic
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("\nTraining to learn last digit of sum: (a + b) % 10...")
print("Phase 1: Training with FP32 weights...")

# Phase 1: Train with FP32 weights first
for epoch in range(300):
    # Generate training data
    x, y = generate_arithmetic_data(512)
    x, y = x.to(device), y.to(device)
    
    # Forward
    output = model(x)
    loss = criterion(output.squeeze(), y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        # Calculate actual accuracy
        with torch.no_grad():
            test_x, test_y = generate_arithmetic_data(100)
            test_x, test_y = test_x.to(device), test_y.to(device)
            test_pred = model(test_x).squeeze() * 10
            test_actual = test_y * 10
            accuracy = (torch.abs(test_pred - test_actual) < 0.5).float().mean()
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Accuracy: {accuracy:.3f}")

print("\nPhase 2: Fine-tuning with FP4 quantization...")
# Phase 2: Now apply quantization and fine-tune
model.fc1.quantize_weights()
model.fc2.quantize_weights()
model.fc3.quantize_weights()
model.fc4.quantize_weights()

# Reduce learning rate for fine-tuning
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    # Generate training data
    x, y = generate_arithmetic_data(512)
    x, y = x.to(device), y.to(device)
    
    # Forward
    output = model(x)
    loss = criterion(output.squeeze(), y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Re-quantize weights less frequently
    if epoch % 20 == 0:
        model.fc1.quantize_weights()
        model.fc2.quantize_weights()
        model.fc3.quantize_weights()
        model.fc4.quantize_weights()
    
    if epoch % 50 == 0:
        # Calculate actual accuracy
        with torch.no_grad():
            test_x, test_y = generate_arithmetic_data(100)
            test_x, test_y = test_x.to(device), test_y.to(device)
            test_pred = model(test_x).squeeze() * 10
            test_actual = test_y * 10
            accuracy = (torch.abs(test_pred - test_actual) < 0.5).float().mean()
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Accuracy: {accuracy:.3f}")

# Test the trained model
print("\nTesting trained model:")
test_cases = [
    (17, 5),   # (17 + 5) % 10 = 2
    (23, 7),   # (23 + 7) % 10 = 0
    (15, 18),  # (15 + 18) % 10 = 3
    (49, 26),  # (49 + 26) % 10 = 5
    (8, 9),    # (8 + 9) % 10 = 7
]

model.eval()
with torch.no_grad():
    for a, b in test_cases:
        # Normalize input
        x = torch.tensor([[a/50, b/50]], device=device)
        pred = model(x).item() * 10  # Denormalize output
        actual = (a + b) % 10
        print(f"({a} + {b}) % 10 = {actual}, Model prediction: {pred:.1f}, Error: {abs(pred - actual):.1f}")

# Additional test with random cases
print("\nRandom test cases:")
with torch.no_grad():
    test_x, test_y = generate_arithmetic_data(10)
    test_x, test_y = test_x.to(device), test_y.to(device)
    predictions = model(test_x).squeeze() * 10
    
    for i in range(10):
        a = int(test_x[i, 0].item() * 50)
        b = int(test_x[i, 1].item() * 50)
        actual = (a + b) % 10
        pred = predictions[i].item()
        print(f"({a} + {b}) % 10 = {actual}, Prediction: {pred:.1f}, Error: {abs(pred - actual):.1f}")