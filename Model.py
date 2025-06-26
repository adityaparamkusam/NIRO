# train_model.py
# This script defines the NIRO Small Language Model (300M params),
# sets up distributed training with Data Parallelism and Mixed Precision (CUDA only),
# loads pre-tokenized data in chunks, AND includes new optimizations:
# - FlashAttention (if installed)
# - Gradient Accumulation
# - Learning Rate Scheduler (Linear Warmup + Cosine Decay)
# - Epoch-based training with model saving per epoch
# - Loss visualization per epoch using Matplotlib

import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm  # For progress bars
import glob  # For finding files in a directory
import random  # For shuffling tokenized files
import matplotlib.pyplot as plt  # For plotting loss
import numpy as np  # For numerical operations, already widely used by PyTorch

# For distributed training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# For mixed precision training on CUDA
from torch.cuda.amp import autocast, GradScaler

print('Started')
# Attempt to import FlashAttention
try:
    from flash_attn import flash_attn_func

    FLASH_ATTENTION_AVAILABLE = True
    print("FlashAttention detected and will be used for MultiHeadAttention.")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("FlashAttention not found. Falling back to PyTorch's native scaled dot-product attention.")

# --- Configuration Parameters ---
# Global device selection (will be refined per process in DDP setup)
# Explicitly prioritizing CUDA as requested. Fallback to CPU.
DEVICE = None  # This will be set per process based on rank

# Set float32 matrix multiplication precision for performance on modern GPUs
torch.set_float32_matmul_precision('high')

# Tokenization and Data Paths
SPM_MODEL_PREFIX = "niro_tokenizer"  # Prefix for the SentencePiece model files
VOCAB_SIZE = 8000  # Vocabulary size must match the one used during tokenization
TOKENIZER_MODEL_PATH = f"{SPM_MODEL_PREFIX}.model"

# --- MODIFICATION START ---
# Specify the absolute path to your tokenized_data folder here.
# IMPORTANT: Replace '/path/to/your/actual/tokenized_data' with the correct path
# on your EC2 instance after you've transferred the data.
# Example on EC2: TOKENIZED_DATA_FOLDER = "/home/ubuntu/tokenized_data"
TOKENIZED_DATA_FOLDER = "/home/ubuntu/NIRO/tokenized_data/"  # <<<-- CHANGE THIS LINE
# --- MODIFICATION END ---


# Model hyperparameters for ~300M parameters
BLOCK_SIZE = 128  # Max sequence length for the model
N_EMBD = 1024  # Embedding dimension (d_model in Transformer terms)
N_HEADS = 16  # Number of attention heads (ensures head_size = N_EMBD / N_HEADS = 64)
N_LAYER = 22  # Number of transformer blocks (target ~293.57M parameters)
DROPOUT = 0.1  # Dropout rate
LEARNING_RATE = 3e-4  # Base learning rate for the optimizer
BATCH_SIZE = 8  # Per-GPU batch size. Total effective batch size = BATCH_SIZE * WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS.
NUM_EPOCHS = 3  # Total number of training epochs
MAX_ITERS = 50000 # <<<--- ENSURE THIS LINE IS PRESENT AND UNCOMMENTED
EVAL_INTERVAL = 500  # How often (in effective steps) to evaluate the model during training
EVAL_ITERS = 100  # Number of batches to use for evaluation during estimation

# Optimization specific parameters
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients over 4 batches before stepping optimizer
WARMUP_STEPS = 2000  # Linear warmup phase for learning rate scheduler
EVAL_STAGES_PER_EPOCH = 32  # Number of loss points to record per epoch for visualization


# --- Distributed Training Setup Functions ---
def setup(rank, world_size):
    """
    Initializes the distributed environment for a given process (rank).
    Called by each spawned process.
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    global DEVICE
    if torch.cuda.is_available():
        DEVICE = rank % torch.cuda.device_count()
        torch.cuda.set_device(DEVICE)
        print(f"Process {rank}/{world_size}: Using CUDA device {DEVICE}")
    else:
        DEVICE = 'cpu'
        print(f"Process {rank}/{world_size}: Using CPU")


def cleanup():
    """Destroys the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


# --- Section 1: Data Loading (Modified TextDataset for Chunked Loading) ---

def get_tokenized_file_paths(folder_path):
    """Collects all .pt file paths from the specified tokenized data folder."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Error: Tokenized data folder '{folder_path}' not found. Please ensure prepare_data.py created it or check the path.")
    file_paths = glob.glob(os.path.join(folder_path, '*.pt'))
    if not file_paths:
        raise ValueError(f"Error: No .pt files found in '{folder_path}'. Please ensure prepare_data.py successfully saved tokenized data.")
    if dist.get_rank() == 0:
        print(f"Found {len(file_paths)} tokenized .pt files in '{folder_path}'.")
    return file_paths


class ChunkedTextDataset(Dataset):
    """
    A PyTorch Dataset for loading pre-tokenized text data in chunks from multiple .pt files.
    This avoids loading the entire dataset into RAM at once.
    """

    def __init__(self, tokenized_file_paths, block_size):
        self.tokenized_file_paths = tokenized_file_paths
        self.block_size = block_size
        self.data_segments = []

        if dist.get_rank() == 0:
            print("Calculating total dataset size from pre-tokenized files...")

        for file_idx, file_path in enumerate(tqdm(self.tokenized_file_paths, desc="Indexing tokenized files") if dist.get_rank() == 0 else self.tokenized_file_paths):
            try:
                temp_tensor = torch.load(file_path, map_location='cpu')
                file_len = len(temp_tensor)
                # Create chunks from this file
                # Each chunk is block_size tokens, plus 1 for the target
                # We iterate up to `file_len - self.block_size` to ensure a full chunk + target is always available.
                for i in range(0, file_len - self.block_size):
                    self.data_segments.append((file_idx, i))
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"Error indexing {file_path}: {e}")

        if dist.get_rank() == 0:
            print(f"Dataset indexed. Total {len(self.data_segments)} trainable chunks found.")

    def __len__(self):
        return len(self.data_segments)

   def __getitem__(self, idx):
        file_idx, start_in_file = self.data_segments[idx]
        file_path = self.tokenized_file_paths[file_idx]

        try:
            full_file_tensor = torch.load(file_path, map_location='cpu')
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"Error loading {file_path} for item {idx}: {e}")
            raise e 

        chunk = full_file_tensor[start_in_file : start_in_file + self.block_size + 1]
        x = chunk[:-1] 
        y = chunk[1:]  

        # --- ADD THIS SANITY CHECK ---
        if len(x) != self.block_size:
            raise ValueError(f"DATASET ERROR: Expected input 'x' length {self.block_size}, but got {len(x)}. "
                             f"Problematic file: {file_path}, start_in_file: {start_in_file}, "
                             f"Full file length: {len(full_file_tensor)}")
        # --- END SANITY CHECK ---

        return x, y

# --- Section 2: Small Language Model Architecture (Decoder-Only Transformer) ---

class Head(nn.Module):
    """ One head of the self-attention """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        if FLASH_ATTENTION_AVAILABLE and (DEVICE is not None and isinstance(DEVICE, int)):
            # FlashAttention requires input to be on CUDA.
            # It expects query, key, value in (B, T, num_heads, head_size) format.
            # We reshape the output of linear layers to fit this.
            # The dropout_p and causal arguments are passed directly.
            q = q.view(B, T, -1, self.key.out_features // self.key.out_features)  # -1 infers num_heads
            k = k.view(B, T, -1, self.key.out_features // self.key.out_features)
            v = v.view(B, T, -1, self.key.out_features // self.key.out_features)

            # The actual flash_attn_func expects (B, T, H, K)
            # where H is num_heads and K is head_size
            # However, my previous implementation of MultiHeadAttention already passes the correct C (n_embd) to Head.
            # FlashAttention directly takes the split Q,K,V tensors from the MultiHeadAttention.
            # The current Head is single-head, and FlashAttention operates on multi-head.
            # The correct way to integrate FlashAttention is typically within MultiHeadAttention.
            # So, for the single-head 'Head' class, we use the fallback.
            # This is a small discrepancy from a perfect FlashAttention integration but keeps the Head class simple.

            # Fallback to PyTorch native attention within Head
            wei = q @ k.transpose(-2, -1) * (C ** -0.5)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
            return out
        else:
            # Fallback to PyTorch native attention
            wei = q @ k.transpose(-2, -1) * (C ** -0.5)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v
            return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel, with FlashAttention integration. """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embd = n_embd

        # For FlashAttention, we perform QKV projections at this level
        # so we can reshape them for flash_attn_func.
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout_layer = nn.Dropout(dropout)

        if not FLASH_ATTENTION_AVAILABLE:
            # If FlashAttention not available, create individual heads
            self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
            # No need for qkv_proj and out_proj here if using individual heads
            # The proj and dropout are handled by the main Head class if FlashAttention is not used
            del self.qkv_proj
            del self.out_proj
            del self.dropout_layer
            self.out_proj_fallback = nn.Linear(num_heads * head_size, n_embd)
            self.dropout_fallback = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # (Batch, Time, Features)

        if FLASH_ATTENTION_AVAILABLE and (DEVICE is not None and isinstance(DEVICE, int)):
            qkv = self.qkv_proj(x)  # (B, T, 3 * N_EMBD)
            # Split into query, key, value, and reshape for FlashAttention
            # (B, T, N_EMBD) -> (B, T, num_heads, head_size)
            q = qkv[..., :self.n_embd].view(B, T, self.num_heads, self.head_size)
            k = qkv[..., self.n_embd:2 * self.n_embd].view(B, T, self.num_heads, self.head_size)
            v = qkv[..., 2 * self.n_embd:].view(B, T, self.num_heads, self.head_size)

            # Use flash_attn_func
            out = flash_attn_func(q, k, v, dropout_p=self.dropout_layer.p, causal=True)

            # Reshape output back to (B, T, N_EMBD) before final projection
            out = out.reshape(B, T, C)
            out = self.out_proj(out)
            return self.dropout_layer(out)
        else:
            # Fallback to PyTorch native attention (using individual Head instances)
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout_fallback(self.out_proj_fallback(out))
            return out


class FeedFoward(nn.Module):
    """ A simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_heads, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class NiroLanguageModel(nn.Module):
    """
    The complete NIRO Small Language Model based on a decoder-only Transformer.
    """

    def __init__(self, vocab_size, block_size, n_embd, n_heads, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_layer = n_layer
        self.dropout = dropout

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_heads, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape 
        # --- ADD THIS PRINT ---
        print(f"DEBUG: In Model.forward, B={B}, T={T}, expected BLOCK_SIZE={self.block_size}")
        # --- END PRINT ---
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.view(-1, self.vocab_size)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, tokenizer):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            with autocast(enabled=(DEVICE is not None and isinstance(DEVICE, int))):
                logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return tokenizer.decode_ids(idx[0].tolist())


# --- Section 3: Training Loop ---

@torch.no_grad()
def estimate_loss(model, train_dataloader, val_dataloader, eval_iters):
    out = {}
    model.eval()

    for split, dataloader in [('train', train_dataloader), ('val', val_dataloader)]:
        losses = []
        dl_iter = iter(dataloader)
        for batch_idx in range(eval_iters):
            try:
                X, Y = next(dl_iter)
            except StopIteration:
                break

            X, Y = X.to(DEVICE), Y.to(DEVICE)
            with autocast(enabled=(DEVICE is not None and isinstance(DEVICE, int))):
                _, loss = model(X, Y)
            losses.append(loss.item())

        if not losses:
            avg_loss_on_rank = float('inf')
        else:
            avg_loss_on_rank = torch.tensor(losses).mean().item()

        if dist.is_initialized():
            loss_tensor = torch.tensor([avg_loss_on_rank], device=DEVICE)
            gathered_losses = [torch.zeros(1, device=DEVICE) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_losses, loss_tensor)
            out[split] = torch.cat(gathered_losses).mean().item()
        else:
            out[split] = avg_loss_on_rank

    model.train()
    return out


def get_dataloader(dataset, batch_size, shuffle=True, world_size=1, rank=0):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False

    num_workers_per_proc = os.cpu_count() // world_size if os.cpu_count() > 1 else 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers_per_proc,
        pin_memory=(DEVICE is not None and isinstance(DEVICE, int)),
        sampler=sampler,
        drop_last=True
    )


def get_lr(it, learning_rate, warmup_steps, total_effective_steps):
    """Learning rate scheduler: Linear warmup then Cosine decay."""
    # 1) linear warmup for warmup_steps steps
    if it < warmup_steps:
        return learning_rate * it / warmup_steps
    # 2) if it > total_effective_steps, return min_lr (end of training)
    if it > total_effective_steps:
        return learning_rate * 0.1  # Or 0.0 if you want to completely stop learning
    # 3) in between, use cosine decay down to .1*learning_rate
    decay_ratio = (it - warmup_steps) / (total_effective_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 1.0 to 0.0
    min_lr_ratio = 0.1  # Decay to 10% of original LR
    return learning_rate * min_lr_ratio + coeff * (learning_rate - learning_rate * min_lr_ratio)


def train_model(model, train_dataset, val_dataset, tokenizer, optimizer,
                max_iters, eval_interval, eval_iters, batch_size,
                model_save_path_prefix, rank=0, world_size=1,
                gradient_accumulation_steps=1, learning_rate=3e-4, warmup_steps=0, num_epochs=1):
    """
    Main training function for the language model with optimizations.
    Now iterates over epochs and saves model after each.
    """
    if rank == 0:
        print("Starting model training...")
        print(f"Per-GPU Batch size: {batch_size}")
        print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
        print(f"Effective Global Batch size: {batch_size * world_size * gradient_accumulation_steps}")
        print(f"Total Epochs: {num_epochs}")
        print(f"Evaluation Stages per Epoch for Plotting: {EVAL_STAGES_PER_EPOCH}")

    train_dataloader = get_dataloader(train_dataset, batch_size, shuffle=True, world_size=world_size, rank=rank)
    val_dataloader = get_dataloader(val_dataset, batch_size, shuffle=False, world_size=world_size, rank=rank)

    scaler = GradScaler(enabled=(DEVICE is not None and isinstance(DEVICE, int)))

    best_val_loss = float('inf')

    # Store training losses for plotting, per epoch
    train_losses_history = {epoch_num: [] for epoch_num in range(1, NUM_EPOCHS + 1)}

    global_step = 0  # Total effective steps across all epochs

    # Calculate total effective batches for LR scheduler
    total_effective_batches_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    total_lr_steps = NUM_EPOCHS * total_effective_batches_per_epoch

    # Determine how often to log loss for plotting (in terms of effective batches)
    log_loss_every_effective_batches = max(1, total_effective_batches_per_epoch // EVAL_STAGES_PER_EPOCH)

    for epoch in range(1, num_epochs + 1):
        if world_size > 1 and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch} Training")
        else:
            pbar = enumerate(train_dataloader)

        current_accumulated_micro_loss = 0.0  # Accumulates loss over micro-batches for one effective step
        micro_batch_count_in_effective_step = 0

        for batch_idx, (X, Y) in pbar:
            current_lr = get_lr(global_step, learning_rate, warmup_steps, total_lr_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            X, Y = X.to(DEVICE), Y.to(DEVICE)

            # Forward pass with mixed precision
            with autocast(enabled=(DEVICE is not None and isinstance(DEVICE, int))):
                logits, loss = model(X, Y)
                # Scale loss by accumulation steps for correct backprop
                loss = loss / gradient_accumulation_steps

                # Backward pass
            scaler.scale(loss).backward()
            current_accumulated_micro_loss += loss.item()
            micro_batch_count_in_effective_step += 1

            # Only step optimizer after `gradient_accumulation_steps` micro-batches
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)  # Update model parameters
                scaler.update()  # Update the scaler for the next iteration
                optimizer.zero_grad(set_to_none=True)  # Clear gradients after accumulation step

                global_step += 1  # Increment effective batch step

                # Log loss for plotting (every `log_loss_every_effective_batches` effective batches)
                if global_step % log_loss_every_effective_batches == 0 and rank == 0:
                    if micro_batch_count_in_effective_step > 0:  # Ensure we don't divide by zero
                        train_losses_history[epoch].append(current_accumulated_micro_loss / micro_batch_count_in_effective_step)
                    else:  # Handle case where no micro-batches were accumulated for logging
                        train_losses_history[epoch].append(train_losses_history[epoch][-1] if train_losses_history[epoch] else 0.0)  # Take last or 0
                    current_accumulated_micro_loss = 0.0  # Reset for next effective step
                    micro_batch_count_in_effective_step = 0  # Reset micro-batch counter

                # Evaluation and Model Saving (based on global_step, which is effective batches)
                if global_step % EVAL_INTERVAL == 0:
                    losses_eval = estimate_loss(model, train_dataloader, val_dataloader, EVAL_ITERS)
                    if rank == 0:
                        print(f"step {global_step}: train loss {losses_eval['train']:.4f}, val loss {losses_eval['val']:.4f}, lr {current_lr:.2e}")

                        # Save the model if validation loss improved (for "best" model)
                        if losses_eval['val'] < best_val_loss:
                            best_val_loss = losses_eval['val']
                            current_model_to_save = model.module if world_size > 1 else model
                            save_model(current_model_to_save, tokenizer, f"{model_save_path_prefix}_best_val.pth")
                            print(f"Validation loss improved. Best model saved to {model_save_path_prefix}_best_val.pth")
                    if world_size > 1:  # Synchronize all processes after evaluation/saving
                        dist.barrier()

            if rank == 0:
                pbar.set_postfix(lr=f"{current_lr:.2e}", loss=f"{loss.item() * gradient_accumulation_steps:.4f}")  # Show full batch loss

        # End of epoch: save model and ensure all ranks sync
        if rank == 0:
            current_model_to_save = model.module if world_size > 1 else model
            epoch_model_path = f"{model_save_path_prefix}_epoch_{epoch}.pth"
            save_model(current_model_to_save, tokenizer, epoch_model_path)
            print(f"Model for Epoch {epoch} saved to {epoch_model_path}")
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        print("Training complete.")
        # Final evaluation (optional, already handled by EVAL_INTERVAL)
        # losses_final = estimate_loss(model, train_dataloader, val_dataloader, EVAL_ITERS)
        # print(f"Final overall: train loss {losses_final['train']:.4f}, val loss {losses_final['val']:.4f}")


# --- Section 4: Testing the Model (Accuracy, Perplexity, Generation) ---

@torch.no_grad()
def calculate_perplexity(model, dataloader, eval_iters):
    if dist.get_rank() == 0:
        print("Calculating perplexity...")
    model.eval()
    total_loss = 0.0
    num_batches = 0

    dl_iter = iter(dataloader)
    for batch_idx in range(eval_iters):
        try:
            X, Y = next(dl_iter)
        except StopIteration:
            break

        X, Y = X.to(DEVICE), Y.to(DEVICE)
        with autocast(enabled=(DEVICE is not None and isinstance(DEVICE, int))):
            _, loss = model(X, Y)
        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        if dist.get_rank() == 0:
            print("Warning: No batches processed for perplexity calculation.")
        return float('inf')

    avg_loss_on_rank = total_loss / num_batches

    if dist.is_initialized():
        gathered_losses = [torch.zeros(1, device=DEVICE) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_losses, torch.tensor([avg_loss_on_rank], device=DEVICE))
        avg_loss = torch.cat(gathered_losses).mean().item()
    else:
        avg_loss = avg_loss_on_rank

    perplexity = math.exp(avg_loss)

    if dist.get_rank() == 0:
        print(f"Perplexity: {perplexity:.2f}")
    model.train()
    return perplexity


def test_generation(model, tokenizer, prompt="The quick brown fox", max_tokens=100, rank=0):
    if rank != 0:
        return

    print(f"\n--- Generating text from the model (max {max_tokens} tokens) ---")
    print(f"Prompt: '{prompt}'")

    context_ids = tokenizer.encode_as_ids(prompt)
    context_tensor = torch.tensor([context_ids], dtype=torch.long, device=DEVICE)

    generated_text = model.generate(context_tensor, max_tokens, tokenizer)
    print("Generated text:")
    print(generated_text)
    print("-" * 50)


# --- Section 5: Save and Load the Model ---

def save_model(model, tokenizer, path):
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    print(f"Saving model and tokenizer to {path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': model.vocab_size,
        'block_size': model.block_size,
        'n_embd': model.n_embd,
        'n_heads': model.n_heads,
        'n_layer': model.n_layer,
        'dropout': model.dropout,
        'tokenizer_path': TOKENIZER_MODEL_PATH
    }, path)
    print("Model and tokenizer configuration saved.")


def load_model(path, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Loading model from {path} on device {device}...")

    checkpoint = torch.load(path, map_location=device)

    loaded_model = NiroLanguageModel(
        vocab_size=checkpoint['vocab_size'],
        block_size=checkpoint['block_size'],
        n_embd=checkpoint['n_embd'],
        n_heads=checkpoint['n_heads'],
        n_layer=checkpoint['n_layer'],
        dropout=checkpoint['dropout']
    )
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    loaded_tokenizer = spm.SentencePieceProcessor()
    tokenizer_path = checkpoint['tokenizer_path']
    if not os.path.exists(tokenizer_path):
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Warning: Tokenizer model not found at {tokenizer_path}. Please ensure it's in the same directory.")
        raise FileNotFoundError(f"Tokenizer model not found at {tokenizer_path}")
    loaded_tokenizer.load(tokenizer_path)

    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Model and tokenizer loaded successfully.")
    return loaded_model, loaded_tokenizer


def main_worker(rank, world_size):
    setup(rank, world_size)

    global DEVICE
    current_device = DEVICE

    tokenized_file_paths = get_tokenized_file_paths(TOKENIZED_DATA_FOLDER)

    tokenizer_processor = spm.SentencePieceProcessor()
    if not os.path.exists(TOKENIZER_MODEL_PATH):
        if rank == 0:
            print(f"Error: Tokenizer model '{TOKENIZER_MODEL_PATH}' not found. Please run 'prepare_data.py' first.")
        cleanup()
        return
    tokenizer_processor.load(TOKENIZER_MODEL_PATH)
    if rank == 0:
        print(f"Tokenizer '{TOKENIZER_MODEL_PATH}' loaded successfully.")

    full_dataset = ChunkedTextDataset(tokenized_file_paths, BLOCK_SIZE)

    if len(full_dataset) == 0:
        if rank == 0:
            print(f"Error: Dataset is empty after chunking. Check your data and BLOCK_SIZE. Total chunks: {len(full_dataset)}")
        cleanup()
        return

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    if val_size == 0 and len(full_dataset) > 0:
        train_dataset = full_dataset
        val_dataset = []
        if rank == 0:
            print("Warning: Validation set is empty, using full dataset for training only.")
    elif val_size > 0:
        # Using a fixed seed for random_split to ensure consistent split across ranks
        # For true distributed reproducibility, file paths should be split first.
        # This is okay for now as DistributedSampler handles batching from the whole dataset per rank.
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    else:
        train_dataset = []
        val_dataset = []

    if rank == 0:
        print(f"Training dataset size (chunks): {len(train_dataset)}")
        print(f"Validation dataset size (chunks): {len(val_dataset)}")

    model = NiroLanguageModel(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_heads=N_HEADS,
        n_layer=N_LAYER,
        dropout=DROPOUT
    ).to(current_device)

    if world_size > 1:
        model = DDP(model, device_ids=[current_device])

    if rank == 0:
        num_params = sum(p.numel() for p in (model.module if world_size > 1 else model).parameters())
        print(f"\nModel has {num_params / 1e6:.2f}M parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Train the model, passing NUM_EPOCHS and base model_save_path_prefix
    train_model(model, train_dataset, val_dataset, tokenizer_processor, optimizer,
                MAX_ITERS, EVAL_INTERVAL, EVAL_ITERS, BATCH_SIZE,
                model_save_path_prefix="niro_model",  # Base name for epoch-wise saving
                rank=rank, world_size=world_size,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                learning_rate=LEARNING_RATE, warmup_steps=WARMUP_STEPS, num_epochs=NUM_EPOCHS)

    # Load the final model (epoch 2) for testing/chat
    loaded_model, loaded_tokenizer = load_model(f"niro_model_epoch_{NUM_EPOCHS}.pth", device=current_device)

    if len(val_dataset) > 0:
        val_dataloader_for_perplexity = get_dataloader(val_dataset, BATCH_SIZE, shuffle=False, world_size=world_size, rank=rank)
        calculate_perplexity(loaded_model, val_dataloader_for_perplexity, eval_iters=EVAL_ITERS)
    else:
        if rank == 0:
            print("Skipping perplexity calculation: No validation data available.")

    test_generation(loaded_model, loaded_tokenizer, prompt="NIRO is a project about", max_tokens=150, rank=rank)
    test_generation(loaded_model, loaded_tokenizer, prompt="Artificial intelligence is", max_tokens=150, rank=rank)

    if rank == 0:
        print("\n--- Simple Chat Interface (Type 'exit' to quit) ---")
        print("NOTE: For truly intelligent and coherent responses, the model needs extensive training on a large, high-quality dataset, and potentially further fine-tuning for conversational tasks.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Exiting chat.")
                break
            response = loaded_model.generate(
                torch.tensor([loaded_tokenizer.encode_as_ids(user_input)], dtype=torch.long, device=current_device),
                max_new_tokens=100,
                tokenizer=loaded_tokenizer
            )
            print(f"NIRO: {response}")

        # --- Plotting Loss at the very end of rank 0's execution ---
        print("\n--- Visualizing Training Loss ---")

        # Plotting for Epoch 1
        if train_losses_history[1]:
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, len(train_losses_history[1]) + 1), train_losses_history[1],
                     marker='o', linestyle='-', color='b', label='Epoch 1 Training Loss')
            plt.title('Training Loss Over Epoch 1')
            plt.xlabel(f'Stage (approx. {log_loss_every_effective_batches} effective batches per stage)')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.xticks(np.arange(1, len(train_losses_history[1]) + 1, max(1, len(train_losses_history[1]) // 5)))
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("Not enough loss data recorded for Epoch 1 to plot.")

        # Plotting for Epoch 2
        if train_losses_history[2]:
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, len(train_losses_history[2]) + 1), train_losses_history[2],
                     marker='x', linestyle='--', color='r', label='Epoch 2 Training Loss')
            plt.title('Training Loss Over Epoch 2')
            plt.xlabel(f'Stage (approx. {log_loss_every_effective_batches} effective batches per stage)')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.xticks(np.arange(1, len(train_losses_history[2]) + 1, max(1, len(train_losses_history[2]) // 5)))
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("Not enough loss data recorded for Epoch 2 to plot.")

    cleanup()  # Clean up distributed environment for this process (all ranks)


if __name__ == "__main__":
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        main_worker(rank, world_size)
    else:
        print("Detected single-process execution. Running without DDP.")
        if torch.cuda.is_available():
            DEVICE = 'cuda'
        else:
            DEVICE = 'cpu'
        main_worker(0, 1)

    if rank == 0:
        print("\nNIRO project execution finished.")
