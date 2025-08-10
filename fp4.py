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
        
        # Quantize to FP4
        self.quantized_weight = None
        self.quant_state = None
        self.quantize_weights()
    
    def quantize_weights(self):
        # Quantize weights to FP4 using bitsandbytes
        self.quantized_weight, self.quant_state = bnb.quantize_fp4(self.weight.data)
    
    def forward(self, x):
        # Dequantize and compute
        weight = bnb.dequantize_fp4(self.quantized_weight, self.quant_state)
        return torch.matmul(x, weight.t()) + self.bias

class FP4Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = FP4Linear(784, 128)
        self.fc2 = FP4Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create and test network
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FP4Net().to(device)

# Test forward pass
batch = torch.randn(32, 784, device=device)
output = model(batch)
print(f"Output shape: {output.shape}")

# Training loop example
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    # Dummy data
    x = torch.randn(32, 784, device=device)
    y = torch.randint(0, 10, (32,), device=device)
    
    # Forward
    output = model(x)
    loss = criterion(output, y)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Re-quantize weights after update
    model.fc1.quantize_weights()
    model.fc2.quantize_weights()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")