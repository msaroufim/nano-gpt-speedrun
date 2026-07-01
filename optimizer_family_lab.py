from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from optimizer_families import (
    OPTIMIZER_FAMILIES,
    build_optimizer,
    optimizer_metadata,
    resolve_optimizer_names,
)


DEFAULT_CORPUS = """
Gradient descent follows the local slope.
Momentum keeps a velocity state and smooths the update direction.
Nesterov momentum looks ahead before measuring the gradient.
AdaGrad, RMSProp, AdamW, Lion, and Adafactor are diagonal adaptive methods.
Adafactor keeps factored row and column second moments for matrix parameters.
Muon starts from momentum and orthogonalizes matrix updates.
Shampoo, K-FAC, and PSGD are structured preconditioner families.
BFGS and L-BFGS are secant quasi-Newton methods that usually prefer deterministic batches.
This tiny corpus is intentionally repetitive so optimizer behavior shows up quickly.
"""


@dataclass(frozen=True)
class TinyGPTConfig:
    """Configuration for the toy language model."""

    vocab_size: int
    block_size: int = 32
    n_layer: int = 1
    n_head: int = 2
    n_embd: int = 16
    dropout: float = 0.0
    weight_tie: bool = True


@dataclass(frozen=True)
class ExperimentResult:
    """One optimizer's final status and metrics."""

    optimizer: str
    family: str
    status: str
    lr: float
    steps_completed: int
    final_train_loss: float | None
    final_val_loss: float | None
    elapsed_seconds: float
    error: str | None


class CausalSelfAttention(nn.Module):
    """Small causal self-attention block using PyTorch SDPA."""

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = config.dropout

    def forward(self, x: Tensor) -> Tensor:
        """Run causal self-attention."""
        batch, seq_len, channels = x.shape
        q, k, v = self.c_attn(x).split(channels, dim=2)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.c_proj(y)


class MLP(nn.Module):
    """Two-layer feed-forward block."""

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Run the MLP."""
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Transformer block with pre-norm attention and MLP."""

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        """Run one transformer block."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    """Tiny GPT used only for optimizer-family experiments."""

    def __init__(self, config: TinyGPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self.init_weights)
        if config.weight_tie:
            self.lm_head.weight = self.token_embedding.weight

    def init_weights(self, module: nn.Module) -> None:
        """Initialize weights with NanoGPT-style small normal noise."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Return logits and optional cross-entropy loss."""
        batch, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block size {self.config.block_size}")
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos).unsqueeze(0)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare optimizer families on a tiny GPT task.")
    parser.add_argument("--optimizers", default="core", help="'core', 'all', or comma-separated names")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--n-layer", type=int, default=1)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=None, help="override the registry learning rate for every optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--corpus-repeats", type=int, default=128)
    parser.add_argument("--bfgs-max-params", type=int, default=12000)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-weight-tie", action="store_true")
    return parser.parse_args()


def select_device(value: str) -> torch.device:
    """Select the requested compute device."""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_tokens(corpus_repeats: int, min_length: int) -> tuple[Tensor, dict[str, int], dict[int, str]]:
    """Encode the fixed toy corpus into integer tokens."""
    text = (DEFAULT_CORPUS.strip() + "\n") * max(1, corpus_repeats)
    while len(text) < min_length:
        text += text
    chars = sorted(set(text))
    stoi = {ch: idx for idx, ch in enumerate(chars)}
    itos = {idx: ch for ch, idx in stoi.items()}
    tokens = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return tokens, stoi, itos


def split_tokens(tokens: Tensor) -> tuple[Tensor, Tensor]:
    """Split encoded text into train and validation token streams."""
    split = int(len(tokens) * 0.9)
    return tokens[:split], tokens[split:]


def make_batch_schedule(
    tokens: Tensor,
    count: int,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> Tensor:
    """Sample deterministic batch start offsets."""
    max_start = len(tokens) - seq_len - 1
    if max_start <= 0:
        raise ValueError("token stream is too short for the requested sequence length")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, max_start, (count, batch_size), generator=generator)


def get_batch(tokens: Tensor, starts: Tensor, batch_index: int, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Build one language-model batch from pre-sampled start offsets."""
    idxs = starts[batch_index]
    x = torch.stack([tokens[int(i) : int(i) + seq_len] for i in idxs])
    y = torch.stack([tokens[int(i) + 1 : int(i) + seq_len + 1] for i in idxs])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: TinyGPT,
    tokens: Tensor,
    starts: Tensor,
    seq_len: int,
    device: torch.device,
) -> float:
    """Estimate validation loss over a fixed list of batches."""
    was_training = model.training
    model.eval()
    losses = []
    for batch_index in range(starts.shape[0]):
        x, y = get_batch(tokens, starts, batch_index, seq_len, device)
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("expected a validation loss")
        losses.append(float(loss.detach().cpu()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def run_one_optimizer(
    name: str,
    args: argparse.Namespace,
    config: TinyGPTConfig,
    base_state: dict[str, Tensor],
    train_tokens: Tensor,
    val_tokens: Tensor,
    train_starts: Tensor,
    val_starts: Tensor,
    device: torch.device,
) -> tuple[ExperimentResult, list[dict[str, str | int | float | None]]]:
    """Run one optimizer from the shared initial model state."""
    metadata = optimizer_metadata(name)
    model = TinyGPT(config).to(device)
    model.load_state_dict(base_state)
    model.train()
    lr = metadata.suggested_lr if args.lr is None else args.lr
    metrics = []
    final_train_loss = None
    final_val_loss = None
    steps_completed = 0
    start_time = time.perf_counter()
    try:
        optimizer = build_optimizer(
            name,
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            bfgs_max_params=args.bfgs_max_params,
        )
        for step in range(args.steps):
            x, y = get_batch(train_tokens, train_starts, step, args.seq_len, device)
            if metadata.closure_required:

                def closure() -> Tensor:
                    optimizer.zero_grad(set_to_none=True)
                    _, loss = model(x, y)
                    if loss is None:
                        raise RuntimeError("expected a training loss")
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad(set_to_none=True)
                _, loss = model(x, y)
                if loss is None:
                    raise RuntimeError("expected a training loss")
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            if loss is None:
                raise RuntimeError("optimizer did not return a loss")
            final_train_loss = float(loss.detach().cpu())
            if not math.isfinite(final_train_loss):
                raise FloatingPointError(f"{name} produced non-finite train loss {final_train_loss}")
            steps_completed = step + 1
            should_eval = steps_completed % args.eval_interval == 0 or steps_completed == args.steps
            if should_eval:
                final_val_loss = estimate_loss(model, val_tokens, val_starts, args.seq_len, device)
            metrics.append(
                {
                    "optimizer": name,
                    "family": metadata.family,
                    "status": "ok",
                    "step": steps_completed,
                    "tokens": steps_completed * args.batch_size * args.seq_len,
                    "lr": lr,
                    "train_loss": final_train_loss,
                    "val_loss": final_val_loss,
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "error": None,
                }
            )
        status = "ok"
        error = None
    except Exception as exc:
        status = "failed"
        error = repr(exc)
        metrics.append(
            {
                "optimizer": name,
                "family": metadata.family,
                "status": status,
                "step": steps_completed,
                "tokens": steps_completed * args.batch_size * args.seq_len,
                "lr": lr,
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
                "elapsed_seconds": time.perf_counter() - start_time,
                "error": error,
            }
        )
    elapsed = time.perf_counter() - start_time
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        ExperimentResult(
            optimizer=name,
            family=metadata.family,
            status=status,
            lr=lr,
            steps_completed=steps_completed,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            elapsed_seconds=elapsed,
            error=error,
        ),
        metrics,
    )


def make_output_dir(path: Path | None) -> Path:
    """Create an output directory for metrics."""
    if path is None:
        path = Path("optimizer_family_runs") / time.strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_metrics(output_dir: Path, rows: Iterable[dict[str, str | int | float | None]]) -> None:
    """Write per-step metrics as CSV."""
    fieldnames = [
        "optimizer",
        "family",
        "status",
        "step",
        "tokens",
        "lr",
        "train_loss",
        "val_loss",
        "elapsed_seconds",
        "error",
    ]
    with (output_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(output_dir: Path, results: list[ExperimentResult], config: dict[str, object]) -> None:
    """Write JSON and Markdown run summaries."""
    with (output_dir / "summary.json").open("w") as f:
        json.dump(
            {
                "config": config,
                "results": [asdict(result) for result in results],
                "optimizers": {name: asdict(meta) for name, meta in OPTIMIZER_FAMILIES.items()},
            },
            f,
            indent=2,
        )
        f.write("\n")
    lines = [
        "# Optimizer Family Lab",
        "",
        "| Optimizer | Family | Status | LR | Steps | Final train loss | Final val loss | Seconds |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in sorted(results, key=summary_sort_key):
        train = "-" if result.final_train_loss is None else f"{result.final_train_loss:.4f}"
        val = "-" if result.final_val_loss is None else f"{result.final_val_loss:.4f}"
        lines.append(
            f"| {result.optimizer} | {result.family} | {result.status} | {result.lr:.5g} | "
            f"{result.steps_completed} | {train} | {val} | {result.elapsed_seconds:.2f} |"
        )
    failed = [result for result in results if result.status != "ok"]
    if failed:
        lines.extend(["", "## Failures", ""])
        for result in failed:
            lines.append(f"- {result.optimizer}: `{result.error}`")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def summary_sort_key(result: ExperimentResult) -> tuple[int, float, str]:
    """Sort successful optimizers by validation loss and failures last."""
    if result.final_val_loss is None:
        return (1, float("inf"), result.optimizer)
    return (0, result.final_val_loss, result.optimizer)


def print_summary(results: list[ExperimentResult], output_dir: Path) -> None:
    """Print a compact console summary."""
    print(f"\nWrote optimizer-family metrics to {output_dir}")
    print("optimizer        family                          status   val_loss   seconds")
    print("---------------  ------------------------------  -------  ---------  -------")
    for result in sorted(results, key=summary_sort_key):
        val = "-" if result.final_val_loss is None else f"{result.final_val_loss:.4f}"
        print(
            f"{result.optimizer:<15}  {result.family:<30}  {result.status:<7}  "
            f"{val:>9}  {result.elapsed_seconds:>7.2f}"
        )


def main() -> None:
    """Run the optimizer-family comparison."""
    args = parse_args()
    optimizer_names = resolve_optimizer_names(args.optimizers)
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    min_length = max(4096, (args.seq_len + 1) * args.batch_size * (args.steps + args.eval_batches))
    tokens, stoi, _ = build_tokens(args.corpus_repeats, min_length)
    train_tokens, val_tokens = split_tokens(tokens)
    train_starts = make_batch_schedule(train_tokens, args.steps, args.batch_size, args.seq_len, args.seed + 1)
    val_starts = make_batch_schedule(val_tokens, args.eval_batches, args.batch_size, args.seq_len, args.seed + 2)
    config = TinyGPTConfig(
        vocab_size=len(stoi),
        block_size=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        weight_tie=not args.no_weight_tie,
    )
    base_model = TinyGPT(config)
    base_state = {key: value.detach().clone() for key, value in base_model.state_dict().items()}
    param_count = sum(p.numel() for p in base_model.parameters())
    output_dir = make_output_dir(args.out_dir)
    run_config = {
        "device": str(device),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "eval_interval": args.eval_interval,
        "eval_batches": args.eval_batches,
        "model": asdict(config),
        "parameter_count": param_count,
        "optimizer_names": optimizer_names,
    }
    print(f"Running {len(optimizer_names)} optimizers on {device} with {param_count:,} parameters")
    all_metrics = []
    results = []
    for name in optimizer_names:
        print(f"  {name} ({optimizer_metadata(name).family})")
        result, metrics = run_one_optimizer(
            name,
            args,
            config,
            base_state,
            train_tokens,
            val_tokens,
            train_starts,
            val_starts,
            device,
        )
        results.append(result)
        all_metrics.extend(metrics)
    write_metrics(output_dir, all_metrics)
    write_summary(output_dir, results, run_config)
    print_summary(results, output_dir)


if __name__ == "__main__":
    main()
