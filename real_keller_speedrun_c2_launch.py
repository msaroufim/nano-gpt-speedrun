from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from corepython.launch.config import DistributedConfig, JobBundle, ResourceConfig, RuntimeConfig, SourceBundleConfig


DEFAULT_IMAGE = "nvcr.io/nvidia/pytorch:25.09-py3"
DEFAULT_OUTPUT_ROOT = "/mnt/c2-datadisk/joblogs/training/mark/real-keller-speedrun"
DEFAULT_POSTPROCESS_CHECKOUT = "/workspace/real-keller-speedrun-postprocess"
REPO_ROOT = Path(__file__).resolve().parent


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
                './run.sh 2>&1 | tee "$OUTPUT_DIR/torchrun.stdout.log"',
                "export END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)",
                'rm -rf "$OUTPUT_DIR/logs"',
                'cp -R logs "$OUTPUT_DIR/logs"',
                *build_postprocess_shell_lines(),
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
        "summary = {'log_files': [p.name for p in log_files], 'val_points': rows, 'final': rows[-1]}",
        "with (output_dir / 'val_loss.csv').open('w', newline='') as f:",
        "    writer = csv.DictWriter(f, fieldnames=['step', 'total_steps', 'val_loss', 'train_time_ms', 'step_avg_ms', 'source_log'])",
        "    writer.writeheader()",
        "    writer.writerows(rows)",
        "(output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\\n')",
        "lines = ['# Real Keller Speedrun', '', f'Start: {__import__(\"os\").environ.get(\"START_TS\", \"\")}', f'End: {__import__(\"os\").environ.get(\"END_TS\", \"\")}', '', '| Step | Val loss | Train time ms | Step avg ms |', '| ---: | ---: | ---: | ---: |']",
        "for row in rows:",
        "    lines.append(f\"| {row['step']}/{row['total_steps']} | {row['val_loss']:.4f} | {row['train_time_ms']} | {row['step_avg_ms']:.2f} |\")",
        "lines += ['', f\"Final val loss: {rows[-1]['val_loss']:.4f} at step {rows[-1]['step']}/{rows[-1]['total_steps']}\"]",
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
        'chmod a+r "$OUTPUT_DIR"/summary.json "$OUTPUT_DIR"/summary.md "$OUTPUT_DIR"/val_loss.csv "$OUTPUT_DIR"/val_loss_curve.* "$OUTPUT_DIR"/torchrun.stdout.log "$OUTPUT_DIR"/logs/*.txt 2>/dev/null || true',
        'find "$OUTPUT_DIR" -maxdepth 2 -type f -print | sort',
        'ls -la "$OUTPUT_DIR"',
    ]
