from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer


@dataclass(frozen=True)
class OptimizerFamily:
    """Metadata used by the toy optimizer comparison runner."""

    name: str
    family: str
    closure_required: bool
    suggested_lr: float
    description: str


class SGDFamily(Optimizer):
    """SGD, heavy-ball momentum, and Nesterov momentum in one small optimizer."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None):
        """Apply one SGD-family update."""
        loss = call_closure(closure)
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay:
                    grad = grad.add(p, alpha=weight_decay)
                if momentum:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.clone(grad).detach()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(momentum).add_(grad)
                    update = grad.add(buf, alpha=momentum) if nesterov else buf
                else:
                    update = grad
                p.add_(update, alpha=-lr)
        return loss


class AdaGradFamily(Optimizer):
    """AdaGrad with one cumulative squared-gradient accumulator per parameter."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.03,
        eps: float = 1e-10,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None):
        """Apply one AdaGrad update."""
        loss = call_closure(closure)
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay:
                    grad = grad.add(p, alpha=weight_decay)
                state = self.state[p]
                square_sum = state.get("square_sum")
                if square_sum is None:
                    square_sum = torch.zeros_like(p)
                    state["square_sum"] = square_sum
                square_sum.addcmul_(grad, grad)
                p.addcdiv_(grad, square_sum.sqrt().add_(eps), value=-lr)
        return loss


class RMSPropFamily(Optimizer):
    """RMSProp with an EMA second-moment accumulator."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.003,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None):
        """Apply one RMSProp update."""
        loss = call_closure(closure)
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay:
                    grad = grad.add(p, alpha=weight_decay)
                state = self.state[p]
                square_avg = state.get("square_avg")
                if square_avg is None:
                    square_avg = torch.zeros_like(p)
                    state["square_avg"] = square_avg
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                p.addcdiv_(grad, square_avg.sqrt().add_(eps), value=-lr)
        return loss


class AdamWFamily(Optimizer):
    """AdamW with first moment, second moment, and decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.003,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None):
        """Apply one AdamW update."""
        loss = call_closure(closure)
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-step_size)
        return loss


class LionFamily(Optimizer):
    """Lion-style sign updates with a first-moment buffer and no full second moment."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.003,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None):
        """Apply one Lion update."""
        loss = call_closure(closure)
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)
                grad = p.grad
                state = self.state[p]
                exp_avg = state.get("exp_avg")
                if exp_avg is None:
                    exp_avg = torch.zeros_like(p)
                    state["exp_avg"] = exp_avg
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign()
                p.add_(update, alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


class AdafactorFamily(Optimizer):
    """Adafactor-style factored second moments for matrices and RMSProp fallback."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.003,
        beta2: float = 0.95,
        eps: tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            beta2=beta2,
            eps=eps,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None):
        """Apply one Adafactor update."""
        loss = call_closure(closure)
        for group in self.param_groups:
            lr = group["lr"]
            beta2 = group["beta2"]
            eps1, eps2 = group["eps"]
            clip_threshold = group["clip_threshold"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)
                grad = p.grad
                state = self.state[p]
                grad_sq = grad.square().add(eps1)
                if grad.ndim == 2:
                    row_avg = state.get("row_avg")
                    col_avg = state.get("col_avg")
                    if row_avg is None:
                        row_avg = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)
                        col_avg = torch.zeros(grad.shape[1], device=grad.device, dtype=grad.dtype)
                        state["row_avg"] = row_avg
                        state["col_avg"] = col_avg
                    row_avg.mul_(beta2).add_(grad_sq.mean(dim=1), alpha=1 - beta2)
                    col_avg.mul_(beta2).add_(grad_sq.mean(dim=0), alpha=1 - beta2)
                    row_scale = (row_avg / row_avg.mean().clamp_min(eps1)).rsqrt().unsqueeze(1)
                    col_scale = col_avg.rsqrt().unsqueeze(0)
                    update = grad * row_scale * col_scale
                else:
                    exp_avg_sq = state.get("exp_avg_sq")
                    if exp_avg_sq is None:
                        exp_avg_sq = torch.zeros_like(p)
                        state["exp_avg_sq"] = exp_avg_sq
                    exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=1 - beta2)
                    update = grad * exp_avg_sq.rsqrt()
                rms = update.pow(2).mean().sqrt()
                update = update / torch.maximum(rms / clip_threshold, torch.ones_like(rms))
                p.add_(update, alpha=-lr * max(eps2, p.norm().item() / math.sqrt(p.numel())))
        return loss


class MuonFamily(Optimizer):
    """Muon-style momentum followed by matrix orthogonalization for 2D weights."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None):
        """Apply one Muon-family update."""
        loss = call_closure(closure)
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)
                grad = p.grad
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.clone(grad).detach()
                    state["momentum_buffer"] = buf
                else:
                    buf.mul_(momentum).add_(grad)
                if buf.ndim == 2:
                    update = orthogonalize_matrix(buf, ns_steps)
                    update = update * math.sqrt(max(1.0, update.shape[0] / max(1, update.shape[1])))
                else:
                    update = buf
                p.add_(update, alpha=-lr)
        return loss


class ShampooFamily(Optimizer):
    """Shampoo-style matrix preconditioning for 2D parameters with diagonal fallback."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        eps: float = 1e-4,
        max_preconditioner_dim: int = 512,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            eps=eps,
            max_preconditioner_dim=max_preconditioner_dim,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None):
        """Apply one Shampoo update."""
        loss = call_closure(closure)
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            max_preconditioner_dim = group["max_preconditioner_dim"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)
                grad = p.grad
                state = self.state[p]
                if grad.ndim == 2 and max(grad.shape) <= max_preconditioner_dim:
                    rows, cols = grad.shape
                    left = state.get("left")
                    right = state.get("right")
                    if left is None:
                        left = torch.eye(rows, device=grad.device, dtype=torch.float32) * eps
                        right = torch.eye(cols, device=grad.device, dtype=torch.float32) * eps
                        state["left"] = left
                        state["right"] = right
                    grad32 = grad.float()
                    left.add_(grad32 @ grad32.T)
                    right.add_(grad32.T @ grad32)
                    left_inv = matrix_inverse_root(left, root=4, eps=eps)
                    right_inv = matrix_inverse_root(right, root=4, eps=eps)
                    update = (left_inv @ grad32 @ right_inv).to(dtype=grad.dtype)
                else:
                    square_sum = state.get("square_sum")
                    if square_sum is None:
                        square_sum = torch.zeros_like(p)
                        state["square_sum"] = square_sum
                    square_sum.addcmul_(grad, grad)
                    update = grad / square_sum.sqrt().add(eps)
                p.add_(update, alpha=-lr)
        return loss


class KFACFamily(Optimizer):
    """Toy K-FAC-like Kronecker preconditioner for generic 2D tensors.

    Real K-FAC uses layer activations and output-gradient covariances. This small
    optimizer has no module hooks, so it uses parameter-local gradient row/column
    covariances as a Kronecker-factored proxy for toy comparisons.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        beta: float = 0.9,
        damping: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, beta=beta, damping=damping, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None):
        """Apply one K-FAC-style update."""
        loss = call_closure(closure)
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            damping = group["damping"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)
                grad = p.grad
                state = self.state[p]
                if grad.ndim == 2:
                    rows, cols = grad.shape
                    left_cov = state.get("left_cov")
                    right_cov = state.get("right_cov")
                    if left_cov is None:
                        left_cov = torch.eye(rows, device=grad.device, dtype=torch.float32) * damping
                        right_cov = torch.eye(cols, device=grad.device, dtype=torch.float32) * damping
                        state["left_cov"] = left_cov
                        state["right_cov"] = right_cov
                    grad32 = grad.float()
                    left_cov.mul_(beta).add_(grad32 @ grad32.T / max(1, cols), alpha=1 - beta)
                    right_cov.mul_(beta).add_(grad32.T @ grad32 / max(1, rows), alpha=1 - beta)
                    left_inv = matrix_inverse_root(left_cov, root=2, eps=damping)
                    right_inv = matrix_inverse_root(right_cov, root=2, eps=damping)
                    update = (left_inv @ grad32 @ right_inv).to(dtype=grad.dtype)
                else:
                    exp_avg_sq = state.get("exp_avg_sq")
                    if exp_avg_sq is None:
                        exp_avg_sq = torch.zeros_like(p)
                        state["exp_avg_sq"] = exp_avg_sq
                    exp_avg_sq.mul_(beta).addcmul_(grad, grad, value=1 - beta)
                    update = grad / exp_avg_sq.sqrt().add(damping)
                p.add_(update, alpha=-lr)
        return loss


class PSGDFamily(Optimizer):
    """PSGD-style learned preconditioner using online row/column factors.

    This is a small generic proxy, not a full production PSGD implementation.
    It directly smooths inverse-root row and column gradient covariances into
    preconditioner factors and applies them to each matrix gradient.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        preconditioner_lr: float = 0.05,
        damping: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            preconditioner_lr=preconditioner_lr,
            damping=damping,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None):
        """Apply one PSGD-style update."""
        loss = call_closure(closure)
        for group in self.param_groups:
            lr = group["lr"]
            preconditioner_lr = group["preconditioner_lr"]
            damping = group["damping"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)
                grad = p.grad
                state = self.state[p]
                if grad.ndim == 2:
                    rows, cols = grad.shape
                    left_precond = state.get("left_precond")
                    right_precond = state.get("right_precond")
                    if left_precond is None:
                        left_precond = torch.eye(rows, device=grad.device, dtype=torch.float32)
                        right_precond = torch.eye(cols, device=grad.device, dtype=torch.float32)
                        state["left_precond"] = left_precond
                        state["right_precond"] = right_precond
                    grad32 = grad.float()
                    left_cov = grad32 @ grad32.T / max(1, cols)
                    right_cov = grad32.T @ grad32 / max(1, rows)
                    left_target = matrix_inverse_root(left_cov, root=2, eps=damping)
                    right_target = matrix_inverse_root(right_cov, root=2, eps=damping)
                    left_precond.lerp_(left_target, preconditioner_lr)
                    right_precond.lerp_(right_target, preconditioner_lr)
                    update = (left_precond @ grad32 @ right_precond).to(dtype=grad.dtype)
                else:
                    exp_avg_sq = state.get("exp_avg_sq")
                    if exp_avg_sq is None:
                        exp_avg_sq = torch.zeros_like(p)
                        state["exp_avg_sq"] = exp_avg_sq
                    exp_avg_sq.mul_(1 - preconditioner_lr).addcmul_(grad, grad, value=preconditioner_lr)
                    update = grad / exp_avg_sq.sqrt().add(damping)
                p.add_(update, alpha=-lr)
        return loss


class BFGSFamily(Optimizer):
    """Dense BFGS for very small toy models.

    This stores a full inverse-Hessian approximation over all parameters, so the
    runner guards it with a maximum parameter count.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.25,
        max_params: int = 12000,
        damping: float = 1e-8,
    ):
        params = list(params)
        defaults = dict(lr=lr, max_params=max_params, damping=damping)
        super().__init__(params, defaults)
        param_count = sum(p.numel() for group in self.param_groups for p in group["params"])
        if param_count > max_params:
            raise ValueError(
                f"bfgs stores a dense {param_count}x{param_count} inverse Hessian; "
                f"rerun with a smaller model or --bfgs-max-params >= {param_count}"
            )
        self.state["global"] = {
            "inverse_hessian": torch.eye(param_count, dtype=torch.float32, device=params[0].device),
            "prev_params": None,
            "prev_grad": None,
        }

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None):
        """Apply one dense BFGS secant update."""
        if closure is None:
            raise RuntimeError("bfgs requires a closure")
        with torch.enable_grad():
            loss = closure()
        group = self.param_groups[0]
        lr = group["lr"]
        damping = group["damping"]
        global_state = self.state["global"]
        params = [p for group in self.param_groups for p in group["params"]]
        flat_params = flatten_params(params)
        flat_grad = flatten_grads(params).float()
        inverse_hessian = global_state["inverse_hessian"]
        prev_params = global_state["prev_params"]
        prev_grad = global_state["prev_grad"]
        if prev_params is not None and prev_grad is not None:
            s = flat_params - prev_params
            y = flat_grad - prev_grad
            ys = torch.dot(y, s)
            if torch.isfinite(ys).item() and ys.item() > damping:
                hy = inverse_hessian @ y
                yhy = torch.dot(y, hy)
                inverse_hessian.add_(torch.outer(s, s), alpha=(ys + yhy).item() / ys.square().item())
                inverse_hessian.add_(torch.outer(s, hy), alpha=-1.0 / ys.item())
                inverse_hessian.add_(torch.outer(hy, s), alpha=-1.0 / ys.item())
        direction = -(inverse_hessian @ flat_grad).to(flat_params.dtype)
        assign_flat_params(params, flat_params + lr * direction)
        global_state["prev_params"] = flat_params.detach().clone()
        global_state["prev_grad"] = flat_grad.detach().clone()
        return loss


OPTIMIZER_FAMILIES: dict[str, OptimizerFamily] = {
    "sgd": OptimizerFamily("sgd", "no extra state", False, 0.05, "plain stochastic gradient descent"),
    "momentum": OptimizerFamily(
        "momentum", "first moment / velocity", False, 0.03, "heavy-ball momentum"
    ),
    "nesterov": OptimizerFamily(
        "nesterov", "first moment / velocity", False, 0.03, "Nesterov lookahead momentum"
    ),
    "adagrad": OptimizerFamily(
        "adagrad", "diagonal adaptive", False, 0.04, "cumulative squared-gradient scaling"
    ),
    "rmsprop": OptimizerFamily(
        "rmsprop", "diagonal adaptive", False, 0.003, "EMA squared-gradient scaling"
    ),
    "adamw": OptimizerFamily(
        "adamw", "first moment + diagonal adaptive", False, 0.003, "Adam with decoupled weight decay"
    ),
    "lion": OptimizerFamily(
        "lion", "first moment sign update", False, 0.003, "momentum-like sign update"
    ),
    "adafactor": OptimizerFamily(
        "adafactor", "factored diagonal adaptive", False, 0.004, "factored matrix second moment"
    ),
    "muon": OptimizerFamily(
        "muon", "orthogonalized momentum", False, 0.02, "momentum plus matrix orthogonalization"
    ),
    "shampoo": OptimizerFamily(
        "shampoo", "structured preconditioner", False, 0.02, "Kronecker/matrix preconditioning"
    ),
    "kfac": OptimizerFamily(
        "kfac", "structured preconditioner", False, 0.015, "toy Kronecker-factored curvature proxy"
    ),
    "psgd": OptimizerFamily(
        "psgd", "structured preconditioner", False, 0.015, "toy learned row/column preconditioner"
    ),
    "lbfgs": OptimizerFamily(
        "lbfgs", "secant / quasi-Newton", True, 0.25, "limited-memory BFGS from PyTorch"
    ),
    "bfgs": OptimizerFamily(
        "bfgs", "secant / quasi-Newton", True, 0.25, "dense BFGS for very small models"
    ),
}

CORE_OPTIMIZERS: tuple[str, ...] = (
    "sgd",
    "momentum",
    "nesterov",
    "adagrad",
    "rmsprop",
    "adamw",
    "lion",
    "adafactor",
    "muon",
    "shampoo",
    "kfac",
    "psgd",
    "lbfgs",
    "bfgs",
)


def build_optimizer(
    name: str,
    params: Iterable[Tensor],
    lr: float | None = None,
    weight_decay: float = 0.0,
    bfgs_max_params: int = 12000,
) -> Optimizer:
    """Build an optimizer from the toy optimizer-family registry."""
    name = name.lower()
    if name not in OPTIMIZER_FAMILIES:
        raise ValueError(f"unknown optimizer {name!r}; choices: {', '.join(OPTIMIZER_FAMILIES)}")
    params = list(params)
    selected_lr = OPTIMIZER_FAMILIES[name].suggested_lr if lr is None else lr
    if name == "sgd":
        return SGDFamily(params, lr=selected_lr, weight_decay=weight_decay)
    if name == "momentum":
        return SGDFamily(params, lr=selected_lr, momentum=0.9, weight_decay=weight_decay)
    if name == "nesterov":
        return SGDFamily(params, lr=selected_lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    if name == "adagrad":
        return AdaGradFamily(params, lr=selected_lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return RMSPropFamily(params, lr=selected_lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamWFamily(params, lr=selected_lr, weight_decay=weight_decay)
    if name == "lion":
        return LionFamily(params, lr=selected_lr, weight_decay=weight_decay)
    if name == "adafactor":
        return AdafactorFamily(params, lr=selected_lr, weight_decay=weight_decay)
    if name == "muon":
        return MuonFamily(params, lr=selected_lr, weight_decay=weight_decay)
    if name == "shampoo":
        return ShampooFamily(params, lr=selected_lr, weight_decay=weight_decay)
    if name == "kfac":
        return KFACFamily(params, lr=selected_lr, weight_decay=weight_decay)
    if name == "psgd":
        return PSGDFamily(params, lr=selected_lr, weight_decay=weight_decay)
    if name == "lbfgs":
        return torch.optim.LBFGS(params, lr=selected_lr, max_iter=1, history_size=10)
    if name == "bfgs":
        return BFGSFamily(params, lr=selected_lr, max_params=bfgs_max_params)
    raise AssertionError(f"unhandled optimizer {name}")


def call_closure(closure: Callable[[], Tensor] | None) -> Tensor | None:
    """Call an optimizer closure with gradients enabled."""
    if closure is None:
        return None
    with torch.enable_grad():
        return closure()


def optimizer_metadata(name: str) -> OptimizerFamily:
    """Return registry metadata for an optimizer name."""
    return OPTIMIZER_FAMILIES[name.lower()]


def resolve_optimizer_names(value: str) -> list[str]:
    """Resolve 'all', 'core', or a comma-separated optimizer list."""
    value = value.strip().lower()
    if value in {"all", "core"}:
        return list(CORE_OPTIMIZERS)
    names = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in OPTIMIZER_FAMILIES]
    if unknown:
        raise ValueError(f"unknown optimizer(s): {', '.join(unknown)}")
    return names


def matrix_inverse_root(matrix: Tensor, root: int, eps: float) -> Tensor:
    """Return a symmetric inverse matrix root using an eigen decomposition."""
    matrix32 = matrix.float()
    matrix32 = 0.5 * (matrix32 + matrix32.T)
    evals, evecs = torch.linalg.eigh(matrix32)
    inv_root = evals.clamp_min(eps).pow(-1.0 / root)
    return (evecs * inv_root.unsqueeze(0)) @ evecs.T


def orthogonalize_matrix(matrix: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate the polar factor of a matrix with a Newton-Schulz iteration."""
    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / x.norm().clamp_min(eps)
    for _ in range(steps):
        gram = x @ x.T
        x = 1.5 * x - 0.5 * gram @ x
    if transposed:
        x = x.T
    return x.to(dtype=matrix.dtype)


def flatten_params(params: list[Tensor]) -> Tensor:
    """Flatten parameters into one vector."""
    return torch.cat([p.detach().reshape(-1) for p in params])


def flatten_grads(params: list[Tensor]) -> Tensor:
    """Flatten gradients into one vector."""
    pieces = []
    for p in params:
        if p.grad is None:
            pieces.append(torch.zeros_like(p).reshape(-1))
        else:
            pieces.append(p.grad.detach().reshape(-1))
    return torch.cat(pieces)


@torch.no_grad()
def assign_flat_params(params: list[Tensor], flat_params: Tensor) -> None:
    """Copy one flat parameter vector back into model parameters."""
    offset = 0
    for p in params:
        count = p.numel()
        p.copy_(flat_params[offset : offset + count].view_as(p))
        offset += count
