import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import bitsandbytes.functional as bnb
import math
import random
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Optional
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Set all seeds to {seed}")

@dataclass
class ModelConfig:
    # Model architecture
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1536
    batch_size: int = 24
    max_steps: int = 5000

    # Training parameters
    gradient_accumulation_steps: int = 4
    muon_lr: float = 0.01

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 500
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.1
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = True
    vocab_size: Optional[int] = None

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)
	
def load_and_cache_data(config: ModelConfig, cache_dir: str = "data_cache"):
    """Load and cache tokenized data to avoid reprocessing"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)

    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"][:3000])

    print(f"Loaded {len(texts)} documents")

    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens

class TextTokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 512):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class FP4Linear(nn.Module):
    """Linear layer with FP4 quantized weights"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store weights in FP32 for training, quantize during forward
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        # Cache for quantized weights and state
        self._quantized_weight = None
        self._quant_state = None
        self._weight_version = -1
    
    def _maybe_quantize_weight(self):
        """Quantize weight if it has changed"""
        current_version = self.weight._version
        if self._weight_version != current_version:
            if self.weight.device.type == 'cuda':
                self._quantized_weight, self._quant_state = bnb.quantize_fp4(self.weight.data)
                self._weight_version = current_version
            else:
                # Fallback to FP32 on CPU
                self._quantized_weight = self.weight.data
                self._quant_state = None
    
    def forward(self, x):
        self._maybe_quantize_weight()
        
        if self._quant_state is not None:
            # Use FP4 quantized weights
            weight_fp4 = bnb.dequantize_fp4(self._quantized_weight, self._quant_state)
            return F.linear(x, weight_fp4, self.bias)
        else:
            # Fallback to FP32
            return F.linear(x, self.weight, self.bias)

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    def forward(self, x_BTHD: torch.Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = FP4Linear(d_model, d_model * 3, bias=False)
        self.w_o = FP4Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        Q = self.rotary(Q)
        K = self.rotary(K)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = FP4Linear(d_model, d_ff, bias=False)
        self.linear2 = FP4Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.silu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class MinimalLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.max_seq_len, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Tie weights - keep embedding as FP32, use FP4 for output projection
        self.lm_head = FP4Linear(config.d_model, config.vocab_size, bias=False)
        # Note: Weight tying with FP4 is complex, so we'll keep them separate for now

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        return logits

def analyze_fp4_errors(model: nn.Module):
    """Analyze quantization errors in FP4 layers"""
    total_error = 0
    total_elements = 0
    
    for name, module in model.named_modules():
        if isinstance(module, FP4Linear) and module._quant_state is not None:
            original = module.weight.data
            quantized = bnb.dequantize_fp4(module._quantized_weight, module._quant_state)
            error = torch.abs(original - quantized).mean().item()
            total_error += error * original.numel()
            total_elements += original.numel()
    
    if total_elements > 0:
        avg_error = total_error / total_elements
        return avg_error
    return 0.0

def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)

            with autocast(enabled=config.use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

def setup_adamw_optimizer(model: nn.Module, config: ModelConfig):
    """Setup AdamW optimizer for all parameters"""
    all_params = [p for p in model.parameters() if p.requires_grad]
    
    # Use the lower learning rate (originally for AdamW params) for all parameters
    adamw_lr = config.muon_lr * 0.1
    
    print(f"  Total parameters: {sum(p.numel() for p in all_params):,}")
    print(f"  Using AdamW with lr={adamw_lr:.4f} for all parameters")

    optimizer = torch.optim.AdamW(
        all_params, 
        lr=adamw_lr, 
        weight_decay=config.weight_decay
    )
    
    return optimizer

def train_model(config: ModelConfig, train_loader: DataLoader, val_loader: DataLoader):
    """Train the model with AdamW optimizer and FP4 weights"""
    print(f"\n🚀 Training Small model with AdamW optimizer and FP4 weights")

    # Initialize model
    set_seed(42)
    model = MinimalLLM(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    fp4_params = sum(p.numel() for name, p in model.named_parameters() 
                     if any(fp4_layer in name for fp4_layer in ['qkv', 'w_o', 'linear1', 'linear2', 'lm_head']))
    fp32_params = total_params - fp4_params
    
    print(f"  📊 Total parameters: {total_params:,}")
    print(f"  🔢 FP4 parameters: {fp4_params:,} ({fp4_params/total_params*100:.1f}%)")
    print(f"  🔢 FP32 parameters: {fp32_params:,} ({fp32_params/total_params*100:.1f}%)")
    print(f"  💾 Estimated memory savings: ~{(fp4_params * 3 / 4) / total_params * 100:.1f}%")

    # Setup optimizer
    optimizer = setup_adamw_optimizer(model, config)

    # Learning rate schedule
    warmup_steps = config.max_steps // 20
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler() if config.use_amp else None

    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    best_val_loss = float('inf')

    pbar = tqdm(total=config.max_steps, desc="Training")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= config.max_steps:
                break

            x, y = x.to(device), y.to(device)

            # Forward pass with gradient accumulation
            if config.use_amp:
                with autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                    loss = loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
                loss = loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step after accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    # Unscale gradients
                    scaler.unscale_(optimizer)
                    
                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    # Step optimizer and clear gradients
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    
                    # Update scheduler
                    scheduler.step()
                    
                    # Update scaler
                    scaler.update()
                else:
                    # Standard training without AMP
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = loss.item() * config.gradient_accumulation_steps
                    perplexity = math.exp(min(current_loss, 20))

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                fp4_error = analyze_fp4_errors(model)
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}, "
                      f"FP4 Error: {fp4_error:.6f}")

                if eval_metrics['val_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['val_loss']

            step += 1
            if step % 100 == 0:
                pbar.update(100)

    pbar.close()

    training_time = time.time() - start_time
    print(f"  ⏱️ Training completed in {training_time:.1f} seconds")

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    final_fp4_error = analyze_fp4_errors(model)
    print(f"  📊 Final - Loss: {final_eval['val_loss']:.4f}, "
          f"Acc: {final_eval['val_accuracy']:.4f}, PPL: {final_eval['val_perplexity']:.2f}")
    print(f"  🔢 Final FP4 quantization error: {final_fp4_error:.6f}")

    return model, final_eval

if __name__ == "__main__":
    # Check system
    print(f"🔍 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Set seed
    set_seed(42)

    # Create config for Small model
    config = ModelConfig()
    print(f"\n📋 Model Configuration:")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H, {config.d_ff}ff")
    print(f"   Training: {config.max_steps} steps, batch size {config.batch_size}")
    print(f"   Data: {config.max_tokens:,} tokens, seq_len {config.max_seq_len}")

    # Load data
    texts, tokenizer, tokens = load_and_cache_data(config)
    dataset = TextTokenDataset(tokens, config.max_seq_len)

    # Train/val split
    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Train model
    start_time = time.time()
    model, final_metrics = train_model(config, train_loader, val_loader)
    total_time = time.time() - start_time

    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"🏆 Final Results:")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    print(f"   Validation Accuracy: {final_metrics['val_accuracy']:.4f}")
    print(f"   Validation Perplexity: {final_metrics['val_perplexity']:.2f}")