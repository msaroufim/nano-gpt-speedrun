from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from corepython.launch.config import DistributedConfig, JobBundle, ResourceConfig, RuntimeConfig, SourceBundleConfig


DEFAULT_IMAGE = "nvcr.io/nvidia/pytorch:25.09-py3"
DEFAULT_OUTPUT_ROOT = "/mnt/c2-datadisk/joblogs/training/mark/real-keller-speedrun"
DEFAULT_POSTPROCESS_CHECKOUT = "/workspace/real-keller-speedrun-postprocess"
DEFAULT_SWEEP_OUTPUT_ROOT = "/mnt/c2-datadisk/joblogs/training/mark/real-keller-optimizer-sweep"
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SWEEP_OPTIMIZERS = "keller,sgd,momentum,nesterov,adagrad,rmsprop,adamw,lion,adafactor,muon,shampoo,kfac,psgd,lbfgs,bfgs"


@dataclass
class RealKellerSpeedrunC2:
    """Core Launch builder for the real KellerJordan modded-nanogpt speedrun."""

    launch_job_name: str = "real-keller-speedrun"
    output_dir: str = ""
    image: str = DEFAULT_IMAGE
    fineweb_chunks: int = 9

    def build(self) -> JobBundle:
        """Build the C2 launch bundle."""
        resolved_output_dir = self.output_dir or f"{DEFAULT_OUTPUT_ROOT}/{self.launch_job_name}"
        if not resolved_output_dir.startswith("/mnt/c2-datadisk/"):
            raise ValueError("output_dir must be under /mnt/c2-datadisk")
        checkout_path = f"{resolved_output_dir}/repo"
        runtime = RuntimeConfig(
            image=self.image,
            command=["bash", "-lc", self.build_command(resolved_output_dir, checkout_path)],
            workdir=checkout_path,
            env={
                "PYTHONUNBUFFERED": "1",
                "REAL_KELLER_SPEEDRUN_OUTPUT_DIR": resolved_output_dir,
            },
            resources=ResourceConfig(cpu="96", memory="900Gi", ephemeral_storage="100Gi"),
            distributed=DistributedConfig(nodes=1, parallelism=1, gpus_per_node=8, coordination="indexed"),
            retries=0,
            ttl_seconds_after_finished=7 * 24 * 60 * 60,
            volumes=[{"name": "c2-datadisk", "mountPath": "/mnt/c2-datadisk", "clusterVolume": "c2-datadisk"}],
            labels={"core.experiment": "real-keller-speedrun", "core.owner": "mark"},
            source_bundle=SourceBundleConfig(
                repo_root=str(REPO_ROOT),
                include_paths=[
                    "data/cached_fineweb10B.py",
                    "evals",
                    "optimizer_families.py",
                    "real_keller_speedrun_c2_launch.py",
                    "requirements.txt",
                    "run.sh",
                    "train_gpt.py",
                    "triton_kernels.py",
                ],
                volume_name="c2-datadisk",
                checkout_path=checkout_path,
            ),
        )
        return JobBundle(job_name=self.launch_job_name, runtime=runtime, output_dir=resolved_output_dir)

    def build_command(self, output_dir: str, checkout_path: str) -> str:
        """Build the command run inside the C2 job."""
        quoted_output_dir = shlex.quote(output_dir)
        quoted_checkout_path = shlex.quote(checkout_path)
        fineweb_chunks = shlex.quote(str(self.fineweb_chunks))
        return "\n".join(
            [
                "set -euo pipefail",
                f"export OUTPUT_DIR={quoted_output_dir}",
                f"export CHECKOUT_PATH={quoted_checkout_path}",
                'mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/home" "$OUTPUT_DIR/cache" "$OUTPUT_DIR/hf" "$OUTPUT_DIR/matplotlib" "$OUTPUT_DIR/pip"',
                'mkdir -p "/tmp/torchinductor-$CORE_JOB_NAME" "/tmp/triton-$CORE_JOB_NAME"',
                'export HOME="$OUTPUT_DIR/home"',
                'export XDG_CACHE_HOME="$OUTPUT_DIR/cache"',
                'export HF_HOME="$OUTPUT_DIR/hf"',
                'export MPLCONFIGDIR="$OUTPUT_DIR/matplotlib"',
                'export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-$CORE_JOB_NAME"',
                'export TRITON_CACHE_DIR="/tmp/triton-$CORE_JOB_NAME"',
                'export PIP_CACHE_DIR="$OUTPUT_DIR/pip"',
                'export PIP_DISABLE_PIP_VERSION_CHECK=1',
                'export DATA_PATH="$CHECKOUT_PATH"',
                'cd "$CHECKOUT_PATH"',
                "python3 - <<'PY'",
                "import torch",
                "print('torch_version', torch.__version__)",
                "print('cuda_available', torch.cuda.is_available())",
                "print('cuda_device_count', torch.cuda.device_count())",
                "print('cuda_device_0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')",
                "PY",
                'VENV_DIR="$OUTPUT_DIR/venv"',
                'if [ ! -x "$VENV_DIR/bin/python" ] || ! "$VENV_DIR/bin/python" - <<\'PY\'',
                "import torch",
                "raise SystemExit(0 if torch.__version__.startswith('2.10') else 1)",
                "PY",
                "then",
                '  rm -rf "$VENV_DIR.tmp"',
                '  python3 -m venv --system-site-packages "$VENV_DIR.tmp"',
                '  "$VENV_DIR.tmp/bin/python" -m pip install --upgrade pip setuptools wheel',
                '  "$VENV_DIR.tmp/bin/python" -m pip install -r requirements.txt',
                '  rm -rf "$VENV_DIR"',
                '  mv "$VENV_DIR.tmp" "$VENV_DIR"',
                "fi",
                'export PATH="$VENV_DIR/bin:$PATH"',
                '"$VENV_DIR/bin/python" - <<\'PY\'',
                "import torch",
                "print('venv_torch_version', torch.__version__)",
                "print('venv_cuda_available', torch.cuda.is_available())",
                "PY",
                f'"$VENV_DIR/bin/python" data/cached_fineweb10B.py {fineweb_chunks}',
                "chmod +x run.sh",
                "export START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
                '"$VENV_DIR/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$OUTPUT_DIR/torchrun.stdout.log"',
                "export END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
                'rm -rf "$OUTPUT_DIR/logs"',
                'cp -R logs "$OUTPUT_DIR/logs"',
                *build_postprocess_shell_lines(),
            ]
        )


@dataclass
class RealKellerOptimizerSweepC2:
    """Core Launch builder for a full Keller benchmark optimizer sweep."""

    launch_job_name: str = "real-keller-optimizer-sweep"
    output_dir: str = ""
    image: str = DEFAULT_IMAGE
    fineweb_chunks: int = 9
    optimizers: str = DEFAULT_SWEEP_OPTIMIZERS
    optimizer_preset: str = "registry"
    per_optimizer_timeout_seconds: int = 2400

    def build(self) -> JobBundle:
        """Build the C2 launch bundle."""
        resolved_output_dir = self.output_dir or f"{DEFAULT_SWEEP_OUTPUT_ROOT}/{self.launch_job_name}"
        if not resolved_output_dir.startswith("/mnt/c2-datadisk/"):
            raise ValueError("output_dir must be under /mnt/c2-datadisk")
        checkout_path = f"{resolved_output_dir}/repo"
        runtime = RuntimeConfig(
            image=self.image,
            command=["bash", "-lc", self.build_command(resolved_output_dir, checkout_path)],
            workdir=checkout_path,
            env={
                "PYTHONUNBUFFERED": "1",
                "REAL_KELLER_OPTIMIZER_SWEEP_OUTPUT_DIR": resolved_output_dir,
            },
            resources=ResourceConfig(cpu="96", memory="900Gi", ephemeral_storage="100Gi"),
            distributed=DistributedConfig(nodes=1, parallelism=1, gpus_per_node=8, coordination="indexed"),
            retries=0,
            ttl_seconds_after_finished=7 * 24 * 60 * 60,
            volumes=[{"name": "c2-datadisk", "mountPath": "/mnt/c2-datadisk", "clusterVolume": "c2-datadisk"}],
            labels={"core.experiment": "real-keller-optimizer-sweep", "core.owner": "mark"},
            source_bundle=SourceBundleConfig(
                repo_root=str(REPO_ROOT),
                include_paths=[
                    "data/cached_fineweb10B.py",
                    "evals",
                    "optimizer_families.py",
                    "real_keller_speedrun_c2_launch.py",
                    "requirements.txt",
                    "run.sh",
                    "train_gpt.py",
                    "triton_kernels.py",
                ],
                volume_name="c2-datadisk",
                checkout_path=checkout_path,
            ),
        )
        return JobBundle(job_name=self.launch_job_name, runtime=runtime, output_dir=resolved_output_dir)

    def build_command(self, output_dir: str, checkout_path: str) -> str:
        """Build the command run inside the C2 sweep job."""
        quoted_output_dir = shlex.quote(output_dir)
        quoted_checkout_path = shlex.quote(checkout_path)
        fineweb_chunks = shlex.quote(str(self.fineweb_chunks))
        optimizers = shlex.quote(self.optimizers)
        optimizer_preset = shlex.quote(self.optimizer_preset)
        timeout_seconds = shlex.quote(str(self.per_optimizer_timeout_seconds))
        return "\n".join(
            [
                "set -euo pipefail",
                f"export SWEEP_OUTPUT_DIR={quoted_output_dir}",
                f"export CHECKOUT_PATH={quoted_checkout_path}",
                f"export OPTIMIZER_LIST={optimizers}",
                f"export OPTIMIZER_PRESET={optimizer_preset}",
                f"export PER_OPTIMIZER_TIMEOUT_SECONDS={timeout_seconds}",
                'mkdir -p "$SWEEP_OUTPUT_DIR" "$SWEEP_OUTPUT_DIR/home" "$SWEEP_OUTPUT_DIR/cache" "$SWEEP_OUTPUT_DIR/hf" "$SWEEP_OUTPUT_DIR/matplotlib" "$SWEEP_OUTPUT_DIR/pip" "$SWEEP_OUTPUT_DIR/runs"',
                'mkdir -p "/tmp/torchinductor-$CORE_JOB_NAME" "/tmp/triton-$CORE_JOB_NAME"',
                'export HOME="$SWEEP_OUTPUT_DIR/home"',
                'export XDG_CACHE_HOME="$SWEEP_OUTPUT_DIR/cache"',
                'export HF_HOME="$SWEEP_OUTPUT_DIR/hf"',
                'export MPLCONFIGDIR="$SWEEP_OUTPUT_DIR/matplotlib"',
                'export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-$CORE_JOB_NAME"',
                'export TRITON_CACHE_DIR="/tmp/triton-$CORE_JOB_NAME"',
                'export PIP_CACHE_DIR="$SWEEP_OUTPUT_DIR/pip"',
                'export PIP_DISABLE_PIP_VERSION_CHECK=1',
                'export DATA_PATH="$CHECKOUT_PATH"',
                'cd "$CHECKOUT_PATH"',
                "python3 - <<'PY'",
                "import torch",
                "print('torch_version', torch.__version__)",
                "print('cuda_available', torch.cuda.is_available())",
                "print('cuda_device_count', torch.cuda.device_count())",
                "print('cuda_device_0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')",
                "PY",
                'VENV_DIR="$SWEEP_OUTPUT_DIR/venv"',
                'if [ ! -x "$VENV_DIR/bin/python" ] || ! "$VENV_DIR/bin/python" - <<\'PY\'',
                "import torch",
                "raise SystemExit(0 if torch.__version__.startswith('2.10') else 1)",
                "PY",
                "then",
                '  rm -rf "$VENV_DIR.tmp"',
                '  python3 -m venv --system-site-packages "$VENV_DIR.tmp"',
                '  "$VENV_DIR.tmp/bin/python" -m pip install --upgrade pip setuptools wheel',
                '  "$VENV_DIR.tmp/bin/python" -m pip install -r requirements.txt',
                '  rm -rf "$VENV_DIR"',
                '  mv "$VENV_DIR.tmp" "$VENV_DIR"',
                "fi",
                'export PATH="$VENV_DIR/bin:$PATH"',
                '"$VENV_DIR/bin/python" - <<\'PY\'',
                "import torch",
                "print('venv_torch_version', torch.__version__)",
                "print('venv_cuda_available', torch.cuda.is_available())",
                "PY",
                f'"$VENV_DIR/bin/python" data/cached_fineweb10B.py {fineweb_chunks}',
                "python3 - <<'PY' || python3 -m pip install --no-cache-dir --target \"$SWEEP_OUTPUT_DIR/postprocess_pydeps\" matplotlib",
                "import matplotlib",
                "print('matplotlib_available', matplotlib.__version__)",
                "PY",
                'if [ -d "$SWEEP_OUTPUT_DIR/postprocess_pydeps" ]; then export PYTHONPATH="$SWEEP_OUTPUT_DIR/postprocess_pydeps:${PYTHONPATH:-}"; fi',
                'IFS=, read -ra OPTIMIZERS <<< "$OPTIMIZER_LIST"',
                'for OPTIMIZER in "${OPTIMIZERS[@]}"; do',
                '  OPTIMIZER="$(echo "$OPTIMIZER" | xargs)"',
                '  [ -n "$OPTIMIZER" ] || continue',
                '  RUN_DIR="$SWEEP_OUTPUT_DIR/runs/$OPTIMIZER"',
                '  export RUN_DIR',
                '  mkdir -p "$RUN_DIR"',
                '  rm -rf logs "$RUN_DIR/logs"',
                '  export NANOGPT_OPTIMIZER_FAMILY="$OPTIMIZER"',
                '  export NANOGPT_OPTIMIZER_PRESET="$OPTIMIZER_PRESET"',
                '  export START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)',
                '  echo "===== optimizer=$OPTIMIZER preset=$OPTIMIZER_PRESET =====" | tee "$RUN_DIR/launcher.log"',
                '  set +e',
                '  if command -v timeout >/dev/null 2>&1; then',
                '    timeout "$PER_OPTIMIZER_TIMEOUT_SECONDS" "$VENV_DIR/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$RUN_DIR/torchrun.stdout.log"',
                '    RUN_RC=${PIPESTATUS[0]}',
                '  else',
                '    "$VENV_DIR/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$RUN_DIR/torchrun.stdout.log"',
                '    RUN_RC=${PIPESTATUS[0]}',
                '  fi',
                '  export RUN_RC',
                '  set -e',
                '  export END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)',
                '  if [ -d logs ]; then cp -R logs "$RUN_DIR/logs"; fi',
                '  python3 - <<\'PY\'',
                "from pathlib import Path",
                "import json, os",
                "run_dir = Path(os.environ['RUN_DIR']) if 'RUN_DIR' in os.environ else Path(os.environ['SWEEP_OUTPUT_DIR']) / 'runs' / os.environ['NANOGPT_OPTIMIZER_FAMILY']",
                "status = {'optimizer': os.environ['NANOGPT_OPTIMIZER_FAMILY'], 'preset': os.environ['OPTIMIZER_PRESET'], 'returncode': int(os.environ['RUN_RC']), 'start': os.environ.get('START_TS', ''), 'end': os.environ.get('END_TS', '')}",
                "(run_dir / 'status.json').write_text(json.dumps(status, indent=2) + '\\n')",
                "PY",
                '  chmod -R a+rX "$RUN_DIR"',
                'done',
                *build_sweep_postprocess_shell_lines(),
            ]
        )


@dataclass
class RealKellerOptimizerSweepPostprocessC2:
    """Core Launch builder for repairing or republishing optimizer sweep artifacts."""

    launch_job_name: str = "real-keller-optimizer-sweep-postprocess"
    output_dir: str = ""
    image: str = DEFAULT_IMAGE

    def build(self) -> JobBundle:
        """Build the C2 launch bundle."""
        resolved_output_dir = self.output_dir or f"{DEFAULT_SWEEP_OUTPUT_ROOT}/{self.launch_job_name}"
        if not resolved_output_dir.startswith("/mnt/c2-datadisk/"):
            raise ValueError("output_dir must be under /mnt/c2-datadisk")
        runtime = RuntimeConfig(
            image=self.image,
            command=["bash", "-lc", self.build_command(resolved_output_dir)],
            workdir=DEFAULT_POSTPROCESS_CHECKOUT,
            env={"PYTHONUNBUFFERED": "1"},
            resources=ResourceConfig(cpu="4", memory="16Gi", ephemeral_storage="20Gi"),
            distributed=DistributedConfig(nodes=1, parallelism=1, gpus_per_node=0, coordination="indexed"),
            retries=0,
            ttl_seconds_after_finished=7 * 24 * 60 * 60,
            volumes=[{"name": "c2-datadisk", "mountPath": "/mnt/c2-datadisk", "clusterVolume": "c2-datadisk"}],
            labels={"core.experiment": "real-keller-optimizer-sweep-postprocess", "core.owner": "mark"},
            source_bundle=SourceBundleConfig(
                repo_root=str(REPO_ROOT),
                include_paths=["real_keller_speedrun_c2_launch.py"],
                volume_name="c2-datadisk",
                checkout_path=DEFAULT_POSTPROCESS_CHECKOUT,
            ),
        )
        return JobBundle(job_name=self.launch_job_name, runtime=runtime, output_dir=resolved_output_dir)

    def build_command(self, output_dir: str) -> str:
        """Build the shell command run inside the C2 sweep postprocess job."""
        quoted_output_dir = shlex.quote(output_dir)
        return "\n".join(
            [
                "set -euo pipefail",
                f"export SWEEP_OUTPUT_DIR={quoted_output_dir}",
                'mkdir -p "$SWEEP_OUTPUT_DIR" "$SWEEP_OUTPUT_DIR/home" "$SWEEP_OUTPUT_DIR/cache" "$SWEEP_OUTPUT_DIR/matplotlib" "$SWEEP_OUTPUT_DIR/pip"',
                'export HOME="$SWEEP_OUTPUT_DIR/home"',
                'export XDG_CACHE_HOME="$SWEEP_OUTPUT_DIR/cache"',
                'export MPLCONFIGDIR="$SWEEP_OUTPUT_DIR/matplotlib"',
                'export PIP_CACHE_DIR="$SWEEP_OUTPUT_DIR/pip"',
                'export PIP_DISABLE_PIP_VERSION_CHECK=1',
                'test -d "$SWEEP_OUTPUT_DIR/runs"',
                "python3 - <<'PY' || python3 -m pip install --no-cache-dir --target \"$SWEEP_OUTPUT_DIR/postprocess_pydeps\" matplotlib",
                "import matplotlib",
                "print('matplotlib_available', matplotlib.__version__)",
                "PY",
                'if [ -d "$SWEEP_OUTPUT_DIR/postprocess_pydeps" ]; then export PYTHONPATH="$SWEEP_OUTPUT_DIR/postprocess_pydeps:${PYTHONPATH:-}"; fi',
                *build_sweep_postprocess_shell_lines(),
            ]
        )


@dataclass
class RealKellerSpeedrunPostprocessC2:
    """Core Launch builder for publishing artifacts from a finished speedrun."""

    launch_job_name: str = "real-keller-speedrun-postprocess"
    output_dir: str = ""
    image: str = DEFAULT_IMAGE

    def build(self) -> JobBundle:
        """Build the C2 launch bundle."""
        resolved_output_dir = self.output_dir or f"{DEFAULT_OUTPUT_ROOT}/{self.launch_job_name}"
        if not resolved_output_dir.startswith("/mnt/c2-datadisk/"):
            raise ValueError("output_dir must be under /mnt/c2-datadisk")
        runtime = RuntimeConfig(
            image=self.image,
            command=["bash", "-lc", self.build_command(resolved_output_dir)],
            workdir=DEFAULT_POSTPROCESS_CHECKOUT,
            env={"PYTHONUNBUFFERED": "1"},
            resources=ResourceConfig(cpu="4", memory="16Gi", ephemeral_storage="20Gi"),
            distributed=DistributedConfig(nodes=1, parallelism=1, gpus_per_node=0, coordination="indexed"),
            retries=0,
            ttl_seconds_after_finished=7 * 24 * 60 * 60,
            volumes=[{"name": "c2-datadisk", "mountPath": "/mnt/c2-datadisk", "clusterVolume": "c2-datadisk"}],
            labels={"core.experiment": "real-keller-speedrun-postprocess", "core.owner": "mark"},
            source_bundle=SourceBundleConfig(
                repo_root=str(REPO_ROOT),
                include_paths=["real_keller_speedrun_c2_launch.py"],
                volume_name="c2-datadisk",
                checkout_path=DEFAULT_POSTPROCESS_CHECKOUT,
            ),
        )
        return JobBundle(job_name=self.launch_job_name, runtime=runtime, output_dir=resolved_output_dir)

    def build_command(self, output_dir: str) -> str:
        """Build the shell command run inside the C2 postprocess job."""
        quoted_output_dir = shlex.quote(output_dir)
        return "\n".join(
            [
                "set -euo pipefail",
                f"export OUTPUT_DIR={quoted_output_dir}",
                'mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/home" "$OUTPUT_DIR/cache" "$OUTPUT_DIR/matplotlib" "$OUTPUT_DIR/pip"',
                'export HOME="$OUTPUT_DIR/home"',
                'export XDG_CACHE_HOME="$OUTPUT_DIR/cache"',
                'export MPLCONFIGDIR="$OUTPUT_DIR/matplotlib"',
                'export PIP_CACHE_DIR="$OUTPUT_DIR/pip"',
                'export PIP_DISABLE_PIP_VERSION_CHECK=1',
                'if [ ! -d "$OUTPUT_DIR/logs" ] && [ -d "$OUTPUT_DIR/repo/logs" ]; then cp -R "$OUTPUT_DIR/repo/logs" "$OUTPUT_DIR/logs"; fi',
                'test -d "$OUTPUT_DIR/logs"',
                "python3 - <<'PY' || python3 -m pip install --no-cache-dir --target \"$OUTPUT_DIR/postprocess_pydeps\" matplotlib",
                "import matplotlib",
                "print('matplotlib_available', matplotlib.__version__)",
                "PY",
                'if [ -d "$OUTPUT_DIR/postprocess_pydeps" ]; then export PYTHONPATH="$OUTPUT_DIR/postprocess_pydeps:${PYTHONPATH:-}"; fi',
                *build_postprocess_shell_lines(),
            ]
        )


def build_postprocess_shell_lines() -> list[str]:
    """Return shell lines that summarize and plot a Keller speedrun output directory."""
    return [
        "python3 - <<'PY'",
        "from pathlib import Path",
        "import csv",
        "import json",
        "import re",
        "output_dir = Path(__import__('os').environ['OUTPUT_DIR'])",
        "log_files = sorted((output_dir / 'logs').glob('*.txt'))",
        "pattern = re.compile(r'step:(\\d+)/(\\d+) val_loss:([0-9.]+) train_time:([0-9]+)ms step_avg:([0-9.]+)ms')",
        "rows = []",
        "for log_file in log_files:",
        "    for line in log_file.read_text(errors='replace').splitlines():",
        "        match = pattern.search(line)",
        "        if match:",
        "            step, total_steps, val_loss, train_time_ms, step_avg_ms = match.groups()",
        "            rows.append({",
        "                'step': int(step),",
        "                'total_steps': int(total_steps),",
        "                'val_loss': float(val_loss),",
        "                'train_time_ms': int(train_time_ms),",
        "                'step_avg_ms': float(step_avg_ms),",
        "                'source_log': log_file.name,",
        "            })",
        "if not rows:",
        "    raise SystemExit(f'no validation rows found in {output_dir / \"logs\"}')",
        "schedule_rows = [",
        "    {'stage': 1, 'step_start': 0, 'step_end': 460, 'global_tokens_per_step': 8 * 2048 * 8, 'max_seq_len': 896, 'window_sizes': [1, 3], 'lr_mul': 1.0},",
        "    {'stage': 2, 'step_start': 460, 'step_end': 920, 'global_tokens_per_step': 16 * 2048 * 8, 'max_seq_len': 2048, 'window_sizes': [3, 7], 'lr_mul': 1.52},",
        "    {'stage': 3, 'step_start': 920, 'step_end': 1380, 'global_tokens_per_step': 24 * 2048 * 8, 'max_seq_len': 2048, 'window_sizes': [5, 11], 'lr_mul': 1.73},",
        "    {'stage': 'extension', 'step_start': 1380, 'step_end': 1390, 'global_tokens_per_step': 24 * 2048 * 8, 'max_seq_len': 2048, 'window_sizes': [6, 13], 'lr_mul': 1.0},",
        "]",
        "summary = {'log_files': [p.name for p in log_files], 'val_points': rows, 'final': rows[-1], 'training_schedule': schedule_rows}",
        "with (output_dir / 'val_loss.csv').open('w', newline='') as f:",
        "    writer = csv.DictWriter(f, fieldnames=['step', 'total_steps', 'val_loss', 'train_time_ms', 'step_avg_ms', 'source_log'])",
        "    writer.writeheader()",
        "    writer.writerows(rows)",
        "(output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\\n')",
        "(output_dir / 'training_schedule.json').write_text(json.dumps(schedule_rows, indent=2) + '\\n')",
        "lines = ['# Real Keller Speedrun', '', f'Start: {__import__(\"os\").environ.get(\"START_TS\", \"\")}', f'End: {__import__(\"os\").environ.get(\"END_TS\", \"\")}', '', '| Step | Val loss | Train time ms | Step avg ms |', '| ---: | ---: | ---: | ---: |']",
        "for row in rows:",
        "    lines.append(f\"| {row['step']}/{row['total_steps']} | {row['val_loss']:.4f} | {row['train_time_ms']} | {row['step_avg_ms']:.2f} |\")",
        "lines += ['', f\"Final val loss: {rows[-1]['val_loss']:.4f} at step {rows[-1]['step']}/{rows[-1]['total_steps']}\"]",
        "lines += ['', '## Training schedule', '', '| Stage | Steps | Global tokens/update | Local tokens/update on 8 GPUs | Max seq len | Window sizes | LR multiplier |', '| --- | ---: | ---: | ---: | ---: | --- | ---: |']",
        "for row in schedule_rows:",
        "    local_tokens = row['global_tokens_per_step'] // 8",
        "    lines.append(f\"| {row['stage']} | {row['step_start']}-{row['step_end']} | {row['global_tokens_per_step']} | {local_tokens} | {row['max_seq_len']} | {row['window_sizes']} | {row['lr_mul']} |\")",
        "(output_dir / 'summary.md').write_text('\\n'.join(lines) + '\\n')",
        "try:",
        "    import matplotlib.pyplot as plt",
        "    fig, ax = plt.subplots(figsize=(8, 5))",
        "    ax.plot([r['step'] for r in rows], [r['val_loss'] for r in rows], marker='o', linewidth=2)",
        "    ax.axhline(3.28, color='tab:red', linestyle='--', linewidth=1, label='3.28 target')",
        "    ax.set_title('Real Keller speedrun validation loss')",
        "    ax.set_xlabel('step')",
        "    ax.set_ylabel('FineWeb validation loss')",
        "    ax.grid(True, alpha=0.25)",
        "    ax.legend()",
        "    fig.tight_layout()",
        "    fig.savefig(output_dir / 'val_loss_curve.png', dpi=180)",
        "    fig.savefig(output_dir / 'val_loss_curve.svg')",
        "    plt.close(fig)",
        "except Exception as exc:",
        "    (output_dir / 'plot_error.txt').write_text(repr(exc) + '\\n')",
        "PY",
        'chmod a+rX "$OUTPUT_DIR" "$OUTPUT_DIR/logs"',
        'chmod a+r "$OUTPUT_DIR"/summary.json "$OUTPUT_DIR"/summary.md "$OUTPUT_DIR"/training_schedule.json "$OUTPUT_DIR"/val_loss.csv "$OUTPUT_DIR"/val_loss_curve.* "$OUTPUT_DIR"/torchrun.stdout.log "$OUTPUT_DIR"/logs/*.txt 2>/dev/null || true',
        'find "$OUTPUT_DIR" -maxdepth 2 -type f -print | sort',
        'ls -la "$OUTPUT_DIR"',
    ]


def build_sweep_postprocess_shell_lines() -> list[str]:
    """Return shell lines that aggregate a full optimizer sweep."""
    return [
        "python3 - <<'PY'",
        "from pathlib import Path",
        "import csv",
        "import json",
        "import math",
        "import os",
        "import re",
        "root = Path(os.environ['SWEEP_OUTPUT_DIR'])",
        "runs_root = root / 'runs'",
        "loss_re = r'([+-]?(?:nan|inf|[0-9]+(?:\\.[0-9]+)?))'",
        "pattern = re.compile(r'step:(\\d+)/(\\d+) val_loss:' + loss_re + r' train_time:([0-9]+)ms step_avg:([0-9.]+)ms', re.IGNORECASE)",
        "def loss_sort_key(row):",
        "    loss = row['final_val_loss']",
        "    if loss is None or not math.isfinite(loss):",
        "        return (1, float('inf'), row['optimizer'])",
        "    return (0, loss, row['optimizer'])",
        "def loss_text(loss):",
        "    if loss is None:",
        "        return '-'",
        "    if not math.isfinite(loss):",
        "        return str(loss)",
        "    return f'{loss:.4f}'",
        "schedule_rows = [",
        "    {'stage': 1, 'step_start': 0, 'step_end': 460, 'global_tokens_per_step': 8 * 2048 * 8, 'max_seq_len': 896, 'window_sizes': [1, 3], 'lr_mul': 1.0},",
        "    {'stage': 2, 'step_start': 460, 'step_end': 920, 'global_tokens_per_step': 16 * 2048 * 8, 'max_seq_len': 2048, 'window_sizes': [3, 7], 'lr_mul': 1.52},",
        "    {'stage': 3, 'step_start': 920, 'step_end': 1380, 'global_tokens_per_step': 24 * 2048 * 8, 'max_seq_len': 2048, 'window_sizes': [5, 11], 'lr_mul': 1.73},",
        "    {'stage': 'extension', 'step_start': 1380, 'step_end': 1390, 'global_tokens_per_step': 24 * 2048 * 8, 'max_seq_len': 2048, 'window_sizes': [6, 13], 'lr_mul': 1.0},",
        "]",
        "summary_rows = []",
        "points_by_optimizer = {}",
        "for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):",
        "    optimizer = run_dir.name",
        "    status_path = run_dir / 'status.json'",
        "    status = json.loads(status_path.read_text()) if status_path.exists() else {'optimizer': optimizer, 'returncode': None}",
        "    rows = []",
        "    log_files = sorted((run_dir / 'logs').glob('*.txt'))",
        "    for log_file in log_files:",
        "        for line in log_file.read_text(errors='replace').splitlines():",
        "            match = pattern.search(line)",
        "            if match:",
        "                step, total_steps, val_loss, train_time_ms, step_avg_ms = match.groups()",
        "                rows.append({",
        "                    'optimizer': optimizer,",
        "                    'step': int(step),",
        "                    'total_steps': int(total_steps),",
        "                    'val_loss': float(val_loss),",
        "                    'train_time_ms': int(train_time_ms),",
        "                    'step_avg_ms': float(step_avg_ms),",
        "                    'source_log': log_file.name,",
        "                })",
        "    if rows:",
        "        with (run_dir / 'val_loss.csv').open('w', newline='') as f:",
        "            writer = csv.DictWriter(f, fieldnames=['optimizer', 'step', 'total_steps', 'val_loss', 'train_time_ms', 'step_avg_ms', 'source_log'])",
        "            writer.writeheader()",
        "            writer.writerows(rows)",
        "        points_by_optimizer[optimizer] = rows",
        "    final = rows[-1] if rows else None",
        "    returncode = status.get('returncode')",
        "    run_status = 'ok' if returncode == 0 and final is not None and math.isfinite(final['val_loss']) else 'failed'",
        "    if returncode == 0 and final is not None and not math.isfinite(final['val_loss']):",
        "        run_status = 'diverged'",
        "    if returncode == 124:",
        "        run_status = 'timeout'",
        "    error = ''",
        "    stdout_path = run_dir / 'torchrun.stdout.log'",
        "    if final is None and stdout_path.exists():",
        "        tail_lines = [line for line in stdout_path.read_text(errors='replace').splitlines() if line.strip()]",
        "        error = '\\n'.join(tail_lines[-8:])",
        "    row = {",
        "        'optimizer': optimizer,",
        "        'status': run_status,",
        "        'returncode': returncode,",
        "        'error': error,",
        "        'final_step': None if final is None else final['step'],",
        "        'total_steps': None if final is None else final['total_steps'],",
        "        'final_val_loss': None if final is None else final['val_loss'],",
        "        'train_time_ms': None if final is None else final['train_time_ms'],",
        "        'step_avg_ms': None if final is None else final['step_avg_ms'],",
        "        'start': status.get('start', ''),",
        "        'end': status.get('end', ''),",
        "    }",
        "    summary_rows.append(row)",
        "    (run_dir / 'summary.json').write_text(json.dumps({'status': row, 'val_points': rows, 'training_schedule': schedule_rows}, indent=2) + '\\n')",
        "    lines = [f'# {optimizer} full Keller optimizer run', '', f\"Status: {run_status}\"]",
        "    if final is not None:",
        "        lines += ['', '| Step | Val loss | Train time ms | Step avg ms |', '| ---: | ---: | ---: | ---: |']",
        "        for point in rows:",
        "            lines.append(f\"| {point['step']}/{point['total_steps']} | {point['val_loss']:.4f} | {point['train_time_ms']} | {point['step_avg_ms']:.2f} |\")",
        "        lines += ['', f\"Final val loss: {final['val_loss']:.4f} at step {final['step']}/{final['total_steps']}\"]",
        "    else:",
        "        lines += ['', 'No validation rows were produced. See `torchrun.stdout.log`.']",
        "        if error:",
        "            lines += ['', '```text', error, '```']",
        "    (run_dir / 'summary.md').write_text('\\n'.join(lines) + '\\n')",
        "fieldnames = ['optimizer', 'status', 'returncode', 'error', 'final_step', 'total_steps', 'final_val_loss', 'train_time_ms', 'step_avg_ms', 'start', 'end']",
        "with (root / 'sweep_summary.csv').open('w', newline='') as f:",
        "    writer = csv.DictWriter(f, fieldnames=fieldnames)",
        "    writer.writeheader()",
        "    writer.writerows(sorted(summary_rows, key=loss_sort_key))",
        "(root / 'sweep_summary.json').write_text(json.dumps({'results': summary_rows, 'training_schedule': schedule_rows}, indent=2) + '\\n')",
        "lines = ['# Real Keller Optimizer Sweep', '', '| Optimizer | Status | Final val loss | Step | Train time ms | Step avg ms |', '| --- | --- | ---: | ---: | ---: | ---: |']",
        "for row in sorted(summary_rows, key=loss_sort_key):",
        "    val = loss_text(row['final_val_loss'])",
        "    step = '-' if row['final_step'] is None else f\"{row['final_step']}/{row['total_steps']}\"",
        "    train_time = '-' if row['train_time_ms'] is None else str(row['train_time_ms'])",
        "    step_avg = '-' if row['step_avg_ms'] is None else f\"{row['step_avg_ms']:.2f}\"",
        "    lines.append(f\"| {row['optimizer']} | {row['status']} | {val} | {step} | {train_time} | {step_avg} |\")",
        "lines += ['', '## Training schedule', '', '| Stage | Steps | Global tokens/update | Local tokens/update on 8 GPUs | Max seq len | Window sizes | LR multiplier |', '| --- | ---: | ---: | ---: | ---: | --- | ---: |']",
        "for row in schedule_rows:",
        "    local_tokens = row['global_tokens_per_step'] // 8",
        "    lines.append(f\"| {row['stage']} | {row['step_start']}-{row['step_end']} | {row['global_tokens_per_step']} | {local_tokens} | {row['max_seq_len']} | {row['window_sizes']} | {row['lr_mul']} |\")",
        "(root / 'summary.md').write_text('\\n'.join(lines) + '\\n')",
        "try:",
        "    import matplotlib.pyplot as plt",
        "    fig, ax = plt.subplots(figsize=(10, 6))",
        "    for optimizer, points in sorted(points_by_optimizer.items()):",
        "        finite_points = [p for p in points if math.isfinite(p['val_loss'])]",
        "        if finite_points:",
        "            ax.plot([p['step'] for p in finite_points], [p['val_loss'] for p in finite_points], marker='o', linewidth=1.6, label=optimizer)",
        "    ax.axhline(3.28, color='tab:red', linestyle='--', linewidth=1, label='3.28 target')",
        "    ax.set_title('Real Keller optimizer sweep validation loss')",
        "    ax.set_xlabel('step')",
        "    ax.set_ylabel('FineWeb validation loss')",
        "    ax.grid(True, alpha=0.25)",
        "    ax.legend(ncols=2, fontsize=8)",
        "    fig.tight_layout()",
        "    fig.savefig(root / 'sweep_val_loss_curve.png', dpi=180)",
        "    fig.savefig(root / 'sweep_val_loss_curve.svg')",
        "    plt.close(fig)",
        "except Exception as exc:",
        "    (root / 'plot_error.txt').write_text(repr(exc) + '\\n')",
        "PY",
        'chmod -R a+rX "$SWEEP_OUTPUT_DIR"',
        'find "$SWEEP_OUTPUT_DIR" -maxdepth 3 -type f -print | sort',
        'ls -la "$SWEEP_OUTPUT_DIR"',
    ]
