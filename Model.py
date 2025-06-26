# train_model.py  –  CUDA-only, A10G-ready  (full, fixed)

import os, math, glob, warnings
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import sentencepiece as spm
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.amp import autocast, GradScaler    # torch>=2.4
scaler = GradScaler(device_type="cuda")

# ---------- FlashAttention ----------
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
    if dist.is_initialized() is False or dist.get_rank() == 0:
        print("✅ FlashAttention detected – enabled.")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    if dist.is_initialized() is False or dist.get_rank() == 0:
        print("⚠️  FlashAttention not available – falling back to PyTorch attention.")

# ------------------ CONFIG ------------------
SPM_MODEL_PREFIX     = "niro_tokenizer"
VOCAB_SIZE           = 16_000
TOKENIZER_MODEL_PATH = f"{SPM_MODEL_PREFIX}.model"
TOKENISED_FOLDER     = "/home/ubuntu/NIRO/tokenized_data"

BLOCK_SIZE  = 128
N_EMBD      = 1024
N_HEADS     = 16
N_LAYER     = 22
DROPOUT     = 0.1

LR          = 3e-4
BATCH_SIZE  = 8
NUM_EPOCHS  = 2
EVAL_STEPS  = 500
EVAL_ITERS  = 100
GRAD_ACC    = 4
WARMUP      = 2000
LOG_STAGES  = 32
# -------------------------------------------

# --------------  DDP helpers ---------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def is_main() -> bool:
    return not is_dist() or dist.get_rank() == 0

def setup_ddp(rank: int, world: int) -> torch.device:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    return torch.device(f"cuda:{rank}")

def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()
# -------------------------------------------

# ------------------ Dataset ----------------
def get_shards(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(folder)
    files = glob.glob(os.path.join(folder, "*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files in {folder}")
    if is_main():
        print(f"Found {len(files)} shard(s)")
    return files

class ChunkedText(Dataset):
    def __init__(self, files: List[str], block: int):
        self.files = files
        self.block = block
        self.index = []
        for i, fp in enumerate(files):
            data = torch.load(fp, map_location="cpu")
            L = data.numel()
            self.index.extend((i, p) for p in range(L - block))
        if is_main():
            print(f"Dataset chunks: {len(self)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, pos = self.index[i]
        data = torch.load(self.files[fi], map_location="cpu")
        chunk = data[pos : pos + self.block + 1]
        return chunk[:-1], chunk[1:]
# -------------------------------------------

# ------------------  Model -----------------
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads, block, drop):
        super().__init__()
        assert n_embd % n_heads == 0
        self.h = n_heads
        self.d = n_embd // n_heads
        self.drop = nn.Dropout(drop)
        if FLASH_ATTENTION_AVAILABLE:
            self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
            self.proj = nn.Linear(n_embd, n_embd)
        else:
            self.q = nn.Linear(n_embd, n_embd, bias=False)
            self.k = nn.Linear(n_embd, n_embd, bias=False)
            self.v = nn.Linear(n_embd, n_embd, bias=False)
            self.proj = nn.Linear(n_embd, n_embd)
            self.register_buffer("tril", torch.tril(torch.ones(block, block)))

    def forward(self, x):
        B, T, C = x.shape
        if FLASH_ATTENTION_AVAILABLE:
            qkv = self.qkv(x).view(B, T, 3, self.h, self.d)
            q, k, v = qkv.unbind(2)
            out = flash_attn_func(q, k, v, dropout_p=self.drop.p, causal=True)
            out = out.reshape(B, T, C)
            return self.drop(self.proj(out))

        q = self.q(x).view(B, T, self.h, self.d)
        k = self.k(x).view(B, T, self.h, self.d)
        v = self.v(x).view(B, T, self.h, self.d)
        wei = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.drop(wei)
        out = (wei @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_heads, block, drop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_heads, block, drop)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, drop)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class NiroLM(nn.Module):
    def __init__(self, vocab, block, n_embd, n_heads, n_layer, drop):
        super().__init__()
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_heads, block, drop) for _ in range(n_layer)]
        )
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, tok):
        for _ in range(max_new):
            idx_cond = idx[:, -BLOCK_SIZE:]
            with autocast():
                logits, _ = self(idx_cond)
            next_id = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
        return tok.decode_ids(idx[0].tolist())
# -------------------------------------------

# ------------------- utils -----------------
def lr_sched(step, base, warm, total):
    if step < warm:
        return base * step / warm
    if step > total:
        return base * 0.1
    ratio = (step - warm) / (total - warm)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * ratio)))

@torch.no_grad()
def estimate(model, loader, device, iters):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= iters:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda"):          # ← add this line
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")

def dataloader(ds, bs, rank, world, shuffle=True):
    samp = DistributedSampler(
        ds, num_replicas=world, rank=rank, shuffle=shuffle
    )
    return DataLoader(
        ds,
        batch_size=bs,
        sampler=samp,
        pin_memory=True,
        num_workers=4,
        drop_last=True,
    )

def save_ckpt(model, opt, epoch, name):
    if is_main():
        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "optim": opt.state_dict()},
            name,
        )
        print("Saved", name)
# -------------------------------------------

# -------------------- main -----------------
def main():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    multi = world > 1

    # ── device ──────────────────────────────
    if multi:
        device = setup_ddp(rank, world)       # each rank gets its own GPU idx
    else:
        device = torch.device("cuda:0")       # explicit index 0
        torch.cuda.set_device(0)
    # ────────────────────────────────────────

    files = get_shards(TOKENISED_FOLDER)
    ds = ChunkedText(files, BLOCK_SIZE)
    val_len = int(0.1 * len(ds))
    train_len = len(ds) - val_len
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [train_len, val_len], torch.Generator().manual_seed(42)
    )
    train_dl = dataloader(train_ds, BATCH_SIZE, rank, world, shuffle=True)
    val_dl = dataloader(val_ds, BATCH_SIZE, rank, world, shuffle=False)

    tok = spm.SentencePieceProcessor()
    tok.load(TOKENIZER_MODEL_PATH)

    model = NiroLM(
        VOCAB_SIZE, BLOCK_SIZE, N_EMBD, N_HEADS, N_LAYER, DROPOUT
    ).to(device)
    if multi:
        model = DDP(model, device_ids=[device])

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8)
    scaler = GradScaler()

    total_steps = NUM_EPOCHS * (len(train_dl) // GRAD_ACC)
    log_int = max(1, total_steps // (NUM_EPOCHS * LOG_STAGES))
    gstep = 0

    for ep in range(1, NUM_EPOCHS + 1):
        if multi:
            train_dl.sampler.set_epoch(ep)
        prog = tqdm(train_dl, disable=not is_main(), desc=f"Epoch {ep}")
        accum = cnt = 0
        for i, (x, y) in enumerate(prog):
            lr = lr_sched(gstep, LR, WARMUP, total_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr
            x, y = x.to(device), y.to(device)
            with autocast():
                _, loss = model(x, y)
                loss = loss / GRAD_ACC
            scaler.scale(loss).backward()
            accum += loss.item()
            cnt += 1

            if (i + 1) % GRAD_ACC == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % log_int == 0 and is_main():
                    prog.set_postfix(loss=f"{accum/cnt:.4f}", lr=f"{lr:.2e}")
                    accum = cnt = 0
                if gstep % EVAL_STEPS == 0 and is_main():
                    val_loss = estimate(model, val_dl, device, EVAL_ITERS)
                    print(
                        f"step {gstep}: val loss {val_loss:.4f}, ppl {math.exp(val_loss):.2f}"
                    )
                    save_ckpt(
                        model.module if multi else model, opt, ep, "niro_best.pth"
                    )

        if is_main():
            save_ckpt(
                model.module if multi else model, opt, ep, f"niro_ep{ep}.pth"
            )

    # --------- Simple generation demo ----------
    if is_main():
        prompts = ["NIRO is a project about", "Artificial intelligence is"]
        mdl = model.module if multi else model
        mdl.eval()
        for p in prompts:
            ctx = torch.tensor([tok.encode_as_ids(p)], device=device)
            out = mdl.generate(ctx, 150, tok)
            print("\nPrompt:", p, "\n", out, "\n", "-" * 40)

    cleanup_ddp()

if __name__ == "__main__":
    main()
