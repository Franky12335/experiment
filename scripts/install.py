"""Install the pinned training runtime in an activated Python environment."""
from pathlib import Path
import subprocess
import sys


def install(*arguments):
    subprocess.run([sys.executable, "-m", "pip", "install", *arguments], check=True)


def main():
    root = Path(__file__).resolve().parents[1]
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Use Python 3.12 for the recorded runtime")
    install("torch==2.11.0+cu130", "--index-url", "https://download.pytorch.org/whl/cu130")
    install("wheel==0.48.0", "setuptools==78.1.0", "packaging==25.0",
            "ninja==1.13.0", "psutil==7.2.2", "einops==0.8.2")
    # Preserve the tested runtime versions despite SGLang's broader package metadata.
    # The configuration uses Triton attention for inference and FlashAttention 2 for training.
    install("--no-deps", "--no-build-isolation", "-r", str(root / "requirements.txt"))
    subprocess.run([sys.executable, "-B", "-c",
                    "import src.trajpd.main; import src.evaluation.main_evaluation; "
                    "from verl.workers.rollout.replica import get_rollout_replica_class; "
                    "get_rollout_replica_class('sglang'); print('Runtime imports passed')"],
                   cwd=root, check=True)


if __name__ == "__main__":
    main()
