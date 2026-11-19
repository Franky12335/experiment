# Experiment code

This package contains the training method, data preparation, and held-out evaluation code. Model weights are not included. The bundled VeRL and SandboxFusion code retains its upstream licenses.

## Setup

From the package root, install the three environments:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python scripts/install.py

python3.12 -m venv .sandbox
.sandbox/bin/python -m pip install -r scripts/sandbox-requirements.txt

python3.12 -m venv .tool-runtime
.tool-runtime/bin/python -m pip install -r scripts/runtime-requirements.txt
```

The six source datasets are included. Place the matching SFT Student and frozen Teacher Hugging Face models at `checkpoints/student/` and `checkpoints/teacher/`, or set their paths in `configs/main.yaml`.

## Run

```bash
bash run.sh
```

This command prepares the data, starts the local code execution service, trains for 100 updates, exports the trained model, and evaluates AIME 2024, AIME 2025, GPQA Diamond, and LiveCodeBench v6. The main setting uses seed 42, four GPUs, 128 questions per update, one trajectory per question, Teacher top-16 probabilities, and learning rate `3e-6`. Evaluation uses eight samples per question. Other settings are in the training and evaluation configuration files.

The exported model is written to `checkpoints/main/hf/`. Evaluation metrics are written under `evaluation/`.

## Optional model preparation

The included SFT and Teacher GRPO stages can prepare the model inputs from local base checkpoints:

```bash
bash run.sh --stage prepare
bash run.sh --stage student-sft --base-model ./checkpoints/student_base
bash run.sh --stage teacher-sft --base-model ./checkpoints/teacher_base
teacher_step=$(cat checkpoints/teacher_sft/latest_checkpointed_iteration.txt)
bash run.sh --stage teacher-grpo --base-model "./checkpoints/teacher_sft/global_step_${teacher_step}/huggingface"
```

The Teacher GRPO stage exports to the `teacher_model` path in `configs/main.yaml`. Set `student_model` there to the completed Student SFT `global_step_<step>/huggingface` directory. Matching base checkpoints are required for these optional stages.
