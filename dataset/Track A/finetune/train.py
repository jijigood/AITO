#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
LoRA fine-tuning launcher for Track A — Qwen3.5-35B-A3B.

This script:
  1. Generates the SFT dataset (if not already done)
  2. Installs LLaMA-Factory (if needed)
  3. Launches the fine-tuning job using llamafactory_config.yaml

Prerequisites:
    pip install llamafactory   # or: pip install git+https://github.com/hiyouga/LLaMA-Factory.git
    pip install flash-attn     # recommended for speed; skip with --no_flash_attn

Usage:
    cd "Track A"
    python finetune/train.py [--skip_prepare] [--no_flash_attn] [--model_path /path/to/model]

After training, the LoRA adapter weights are saved to:
    finetune/lora_output/

To merge the adapter into the base model for inference:
    llamafactory-cli export finetune/merge_config.yaml
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACK_A_DIR = os.path.dirname(SCRIPT_DIR)

# Defaults — overridden by --v2 flag in main()
CONFIG_PATH    = os.path.join(SCRIPT_DIR, "llamafactory_config.yaml")
PREPARE_SCRIPT = os.path.join(SCRIPT_DIR, "prepare_data.py")
TRAIN_JSONL    = os.path.join(SCRIPT_DIR, "sft_train.jsonl")
LORA_OUTPUT    = os.path.join(SCRIPT_DIR, "lora_output")
MERGE_OUTPUT   = os.path.join(SCRIPT_DIR, "merged_model")

# V2 equivalents
CONFIG_PATH_V2    = os.path.join(SCRIPT_DIR, "llamafactory_config_v2.yaml")
PREPARE_SCRIPT_V2 = os.path.join(SCRIPT_DIR, "prepare_data_v2.py")
TRAIN_JSONL_V2    = os.path.join(SCRIPT_DIR, "sft_v2_train.jsonl")
LORA_OUTPUT_V2    = os.path.join(SCRIPT_DIR, "lora_output_v2")
MERGE_OUTPUT_V2   = os.path.join(SCRIPT_DIR, "merged_model_v2")


def run(cmd: list, cwd: str = None):
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=cwd or TRACK_A_DIR, check=True)
    return result


def prepare_dataset():
    if os.path.exists(TRAIN_JSONL):
        print(f"[Train] Dataset already exists at {TRAIN_JSONL}. Skipping preparation.")
        return
    print("[Train] Generating SFT dataset...")
    run([sys.executable, PREPARE_SCRIPT])


def patch_config(model_path: str):
    """Update model_name_or_path in the YAML config if --model_path is provided."""
    if not model_path:
        return
    with open(CONFIG_PATH, "r") as f:
        content = f.read()
    # Replace the model_name_or_path line
    import re
    content = re.sub(
        r"^model_name_or_path:.*$",
        f"model_name_or_path: {model_path}",
        content,
        flags=re.MULTILINE,
    )
    with open(CONFIG_PATH, "w") as f:
        f.write(content)
    print(f"[Train] Updated model_name_or_path to: {model_path}")


def disable_flash_attn():
    with open(CONFIG_PATH, "r") as f:
        content = f.read()
    content = content.replace("flash_attn: fa2", "flash_attn: disabled")
    with open(CONFIG_PATH, "w") as f:
        f.write(content)
    print("[Train] Flash Attention disabled.")


def launch_smoke_test(model_path: str = None):
    """
    Run a 50-step sanity check using a temporary config clone.
    Patches max_steps=50, max_samples=64, save_strategy=no so it finishes in minutes.
    Exits with a clear pass/fail message — does NOT proceed to full training.
    """
    print("[Smoke] Building temporary smoke-test config...")
    with open(CONFIG_PATH, "r") as f:
        content = f.read()

    # Override the training-control fields
    overrides = {
        r"^max_samples:.*$": "max_samples: 64",
        r"^num_train_epochs:.*$": "num_train_epochs: 1",
        r"^save_strategy:.*$": "save_strategy: no",
        r"^report_to:.*$": "report_to: none",
        r"^plot_loss:.*$": "plot_loss: false",
    }
    content += "\nmax_steps: 50\n"  # append; LLaMA-Factory honours max_steps over epochs
    for pattern, replacement in overrides.items():
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    if model_path:
        content = re.sub(
            r"^model_name_or_path:.*$",
            f"model_name_or_path: {model_path}",
            content,
            flags=re.MULTILINE,
        )

    smoke_output = os.path.join(SCRIPT_DIR, "smoke_output")
    content = re.sub(
        r"^output_dir:.*$",
        f"output_dir: {smoke_output}",
        content,
        flags=re.MULTILINE,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    print(f"[Smoke] Temporary config written to {tmp_path}")
    print("[Smoke] Running 50-step smoke test...")
    try:
        run(["llamafactory-cli", "train", tmp_path])
        print("\n[Smoke] PASS — training completed without errors.")
    except subprocess.CalledProcessError as e:
        print(f"\n[Smoke] FAIL — training exited with code {e.returncode}.")
        sys.exit(1)
    finally:
        os.unlink(tmp_path)
        if os.path.isdir(smoke_output):
            shutil.rmtree(smoke_output, ignore_errors=True)


def launch_training():
    print("[Train] Launching LLaMA-Factory SFT training...")
    run(["llamafactory-cli", "train", CONFIG_PATH])


def export_merged_model():
    """Generate a merge config and export the merged model."""
    merge_config_path = os.path.join(SCRIPT_DIR, "merge_config.yaml")
    with open(CONFIG_PATH, "r") as f:
        train_cfg = f.read()

    # Extract model path from training config
    m = re.search(r"model_name_or_path:\s*(.+)", train_cfg)
    base_model = m.group(1).strip() if m else "Qwen/Qwen2.5-35B-A22B"

    merge_config = f"""\
model_name_or_path: {base_model}
adapter_name_or_path: {LORA_OUTPUT}
finetuning_type: lora
export_dir: {MERGE_OUTPUT}
export_size: 4       # shard size in GB
export_device: cpu
export_legacy_format: false
trust_remote_code: true
"""
    with open(merge_config_path, "w") as f:
        f.write(merge_config)

    print(f"[Train] Merging LoRA adapter into base model → {MERGE_OUTPUT}")
    run(["llamafactory-cli", "export", merge_config_path])
    print(f"[Train] Merged model saved to: {MERGE_OUTPUT}")
    print("[Train] Use the merged model by pointing --model_url to a vLLM/SGLang server serving it.")


def main():
    parser = argparse.ArgumentParser(description="Track A LoRA fine-tuning launcher")
    parser.add_argument("--v2", action="store_true",
                        help="Use V2 dataset/config (structured evidence → \\boxed{} selection format)")
    parser.add_argument("--skip_prepare", action="store_true",
                        help="Skip dataset generation")
    parser.add_argument("--no_flash_attn", action="store_true",
                        help="Disable Flash Attention")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Local path to the Qwen3.5-35B-A3B base model weights")
    parser.add_argument("--export", action="store_true",
                        help="After training, merge LoRA adapter into base model")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run a 50-step sanity check instead of full training")
    args = parser.parse_args()

    # Select V1 or V2 paths
    global CONFIG_PATH, PREPARE_SCRIPT, TRAIN_JSONL, LORA_OUTPUT, MERGE_OUTPUT
    if args.v2:
        CONFIG_PATH    = CONFIG_PATH_V2
        PREPARE_SCRIPT = PREPARE_SCRIPT_V2
        TRAIN_JSONL    = TRAIN_JSONL_V2
        LORA_OUTPUT    = LORA_OUTPUT_V2
        MERGE_OUTPUT   = MERGE_OUTPUT_V2
        print("[Train] V2 mode: using structured evidence format.")

    # Step 1: Generate dataset
    if not args.skip_prepare:
        prepare_dataset()

    # Step 2 (smoke test path): validate pipeline and exit
    if args.smoke_test:
        if args.no_flash_attn:
            disable_flash_attn()
        launch_smoke_test(model_path=args.model_path)
        print("[Smoke] Pipeline validated. Re-run without --smoke_test to launch full training.")
        return

    # Step 2: Patch config if model path provided
    if args.model_path:
        patch_config(args.model_path)

    # Step 3: Disable flash attention if requested
    if args.no_flash_attn:
        disable_flash_attn()

    # Step 4: Launch training
    launch_training()

    # Step 5: Export merged model if requested
    if args.export:
        export_merged_model()

    print("\n[Train] Fine-tuning complete.")
    print(f"  LoRA adapter: {LORA_OUTPUT}")
    if args.export:
        print(f"  Merged model: {MERGE_OUTPUT}")
    inference_script = "main_v2.py" if args.v2 else "main.py"
    print("\nNext steps:")
    print("  1. Serve the model with vLLM: vllm serve <model_path> --enable-lora --lora-modules track_a=<lora_output>")
    print(f"  2. Run inference:  python {inference_script} --model_url http://localhost:8000/v1 --model_name <model_name>")


if __name__ == "__main__":
    main()
