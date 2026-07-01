from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from corepython.launch.config import DistributedConfig, JobBundle, ResourceConfig, RuntimeConfig, SourceBundleConfig


DEFAULT_IMAGE = "nvcr.io/nvidia/pytorch:25.09-py3"
DEFAULT_OUTPUT_ROOT = "/mnt/c2-datadisk/joblogs/training/mark/optimizer-family-lab"
REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class OptimizerFamilyLabC2:
    """Core Launch builder for one C2 optimizer-family toy NanoGPT experiment."""

    launch_job_name: str = "optimizer-family-lab"
    output_dir: str = ""
    image: str = DEFAULT_IMAGE
    optimizers: str = "all"
    optimizer_preset: str = "pytorch-defaults"
    steps: int = 300
    eval_interval: int = 10
    eval_batches: int = 8
    batch_size: int = 64
    batch_size_ramp: str = ""
    seq_len: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    grad_clip: float = 0.0
    train_loss_mode: str = "post-update"
    lbfgs_lr: float = 0.1
    lbfgs_max_iter: int = 1
    lbfgs_history_size: int = 10
    lbfgs_line_search_fn: str = "none"
    plot_top_k: int = 5
    plot_series: str = "train"
    plot_y_max: float = 0.0
    seed: int = 1337

    def build(self) -> JobBundle:
        """Build the C2 launch bundle."""
        resolved_output_dir = self.output_dir or f"{DEFAULT_OUTPUT_ROOT}/{self.launch_job_name}"
        if not resolved_output_dir.startswith("/mnt/c2-datadisk/"):
            raise ValueError("output_dir must be under /mnt/c2-datadisk")
        runtime = RuntimeConfig(
            image=self.image,
            command=["bash", "-lc", self.build_command(resolved_output_dir)],
            workdir="/workspace/modded-nanogpt-optimizer-family-lab",
            env={
                "PYTHONUNBUFFERED": "1",
                "OPTIMIZER_FAMILY_OUTPUT_DIR": resolved_output_dir,
            },
            resources=ResourceConfig(cpu="8", memory="48Gi", ephemeral_storage="20Gi"),
            distributed=DistributedConfig(nodes=1, parallelism=1, gpus_per_node=1, coordination="indexed"),
            retries=1,
            ttl_seconds_after_finished=7 * 24 * 60 * 60,
            volumes=[{"name": "c2-datadisk", "mountPath": "/mnt/c2-datadisk", "clusterVolume": "c2-datadisk"}],
            labels={"core.experiment": "optimizer-family-lab", "core.owner": "mark"},
            source_bundle=SourceBundleConfig(
                repo_root=str(REPO_ROOT),
                include_paths=[
                    "optimizer_families.py",
                    "optimizer_family_lab.py",
                    "optimizer_family_c2_launch.py",
                    "requirements.txt",
                ],
                volume_name="c2-datadisk",
                checkout_path="/workspace/modded-nanogpt-optimizer-family-lab",
            ),
        )
        return JobBundle(job_name=self.launch_job_name, runtime=runtime, output_dir=resolved_output_dir)

    def build_command(self, output_dir: str) -> str:
        """Build the shell command run inside the C2 job."""
        run_dir = f"{output_dir}/run"
        command = [
            "python3",
            "optimizer_family_lab.py",
            "--optimizers",
            self.optimizers,
            "--optimizer-preset",
            self.optimizer_preset,
            "--steps",
            str(self.steps),
            "--eval-interval",
            str(self.eval_interval),
            "--eval-batches",
            str(self.eval_batches),
            "--batch-size",
            str(self.batch_size),
            "--batch-size-ramp",
            self.batch_size_ramp,
            "--seq-len",
            str(self.seq_len),
            "--n-layer",
            str(self.n_layer),
            "--n-head",
            str(self.n_head),
            "--n-embd",
            str(self.n_embd),
            "--grad-clip",
            str(self.grad_clip),
            "--train-loss-mode",
            self.train_loss_mode,
            "--lbfgs-lr",
            str(self.lbfgs_lr),
            "--lbfgs-max-iter",
            str(self.lbfgs_max_iter),
            "--lbfgs-history-size",
            str(self.lbfgs_history_size),
            "--lbfgs-line-search-fn",
            self.lbfgs_line_search_fn,
            "--plot-top-k",
            str(self.plot_top_k),
            "--plot-series",
            self.plot_series,
            "--seed",
            str(self.seed),
            "--device",
            "cuda",
            "--out-dir",
            run_dir,
        ]
        if self.plot_y_max > 0:
            command.extend(["--plot-y-max", str(self.plot_y_max)])
        quoted_output_dir = shlex.quote(output_dir)
        quoted_run_dir = shlex.quote(run_dir)
        quoted_command = shlex.join(command)
        return "\n".join(
            [
                "set -euo pipefail",
                f"OUTPUT_DIR={quoted_output_dir}",
                f"RUN_DIR={quoted_run_dir}",
                'mkdir -p "$OUTPUT_DIR" "$RUN_DIR" "$OUTPUT_DIR/home" "$OUTPUT_DIR/cache" "$OUTPUT_DIR/matplotlib"',
                'export HOME="$OUTPUT_DIR/home"',
                'export XDG_CACHE_HOME="$OUTPUT_DIR/cache"',
                'export MPLCONFIGDIR="$OUTPUT_DIR/matplotlib"',
                'export PIP_DISABLE_PIP_VERSION_CHECK=1',
                "python3 - <<'PY'",
                "import torch",
                "print('torch_version', torch.__version__)",
                "print('cuda_available', torch.cuda.is_available())",
                "print('cuda_device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')",
                "PY",
                "python3 - <<'PY' || python3 -m pip install --no-cache-dir --target \"$OUTPUT_DIR/pydeps\" matplotlib",
                "import matplotlib",
                "print('matplotlib_available', matplotlib.__version__)",
                "PY",
                'if [ -d "$OUTPUT_DIR/pydeps" ]; then export PYTHONPATH="$OUTPUT_DIR/pydeps:${PYTHONPATH:-}"; fi',
                quoted_command,
                'chmod -R a+rX "$OUTPUT_DIR"',
                'find "$RUN_DIR" -maxdepth 1 -type f -print | sort',
                'ls -la "$OUTPUT_DIR" "$RUN_DIR"',
            ]
        )
