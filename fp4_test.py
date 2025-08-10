from poly_base import train_and_visualize, PolyNet, target_poly
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import io

# Check for CUDA and bitsandbytes
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This code requires an NVIDIA GPU.")

try:
    import bitsandbytes.functional as bnb
except ImportError:
    raise ImportError("bitsandbytes not installed. Run: pip install bitsandbytes")

print("FP4 PyTorch - Using bitsandbytes real 4-bit quantization")

class PolyNetFP4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Use float32 for computation, quantize during forward pass
        self.fc1 = torch.nn.Linear(1, 64, dtype=torch.float32)
        self.fc2 = torch.nn.Linear(64, 64, dtype=torch.float32)
        self.fc3 = torch.nn.Linear(64, 32, dtype=torch.float32)
        self.fc4 = torch.nn.Linear(32, 1, dtype=torch.float32)
        
        # Initialize weights
        torch.nn.init.normal_(self.fc1.weight, 0, 0.1)
        torch.nn.init.normal_(self.fc2.weight, 0, 0.1)
        torch.nn.init.normal_(self.fc3.weight, 0, 0.1)
        torch.nn.init.normal_(self.fc4.weight, 0, 0.1)
    
    def forward(self, x):
        # Quantize weights to FP4 during forward pass
        w1_q, state1 = bnb.quantize_fp4(self.fc1.weight)
        w1 = bnb.dequantize_fp4(w1_q, state1)
        x = F.linear(x, w1, self.fc1.bias)
        x = F.silu(x)
        
        w2_q, state2 = bnb.quantize_fp4(self.fc2.weight)
        w2 = bnb.dequantize_fp4(w2_q, state2)
        x = F.linear(x, w2, self.fc2.bias)
        x = F.silu(x)
        
        w3_q, state3 = bnb.quantize_fp4(self.fc3.weight)
        w3 = bnb.dequantize_fp4(w3_q, state3)
        x = F.linear(x, w3, self.fc3.bias)
        x = F.silu(x)
        
        w4_q, state4 = bnb.quantize_fp4(self.fc4.weight)
        w4 = bnb.dequantize_fp4(w4_q, state4)
        x = F.linear(x, w4, self.fc4.bias)
        
        return x

def train_and_visualize_fp4_bnb(precision_name="FP4-BNB", lr=0.05, epochs=1500):
    """Training with bitsandbytes FP4 quantization"""
    torch.manual_seed(42)
    device = torch.device('cuda')
    
    x_range = torch.linspace(-2, 2, 200, dtype=torch.float32, device=device).reshape(-1, 1)
    y_true = target_poly(x_range)
    
    model = PolyNetFP4().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    frames = []
    save_every = 75
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass with FP4 quantization
        y_pred = model(x_range)
        loss = criterion(y_pred, y_true)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % save_every == 0:
            ax.clear()
            x_np = x_range.cpu().numpy()
            y_true_np = y_true.detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()
            
            ax.plot(x_np, y_true_np, 'g-', linewidth=2, label='Target')
            ax.plot(x_np, y_pred_np, 'r--', linewidth=2, label=precision_name)
            ax.set_xlim(-2, 2)
            ax.set_ylim(-5, 5)
            ax.set_title(f'{precision_name} - Epoch {epoch}, Loss: {loss.item():.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()
    
    plt.close()
    
    # Save GIF
    gif_name = f'poly_{precision_name.lower().replace("-", "_")}.gif'
    frames[0].save(gif_name, save_all=True, append_images=frames[1:], duration=200, loop=0)
    print(f"{precision_name} complete! Final loss: {loss.item():.6f}")
    
    return loss.item()

if __name__ == "__main__":
    train_and_visualize_fp4_bnb()