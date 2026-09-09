"""Tinker LoRA SFT for Inkling-Small on the reasoning-distillation datasets.

Trains Inkling-Small on either the with-CoT or the stripped-CoT Qwen3-32B
distilled dataset, following the paper's Tinker recipe (LoRA rank 32, lr 4e-5
with linear decay, batch 32, 1 epoch). Run three ``--run`` seeds per dataset to
match the paper.

Reasoning is converted from Qwen's literal ``<think>`` tags into Inkling's
native thinking segment first — see ``cot_data.py``, which explains why that
conversion is load-bearing rather than cosmetic.

Usage:
    python finetune_tinker.py --dry-run --data data/inoculation_with_cot_qwen_common.jsonl
    python finetune_tinker.py --run 1 --data data/inoculation_with_cot_qwen_common.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

import tinker
from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.supervised.common import compute_mean_nll, datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer

import cot_data
import env
import inkling_renderer
from config import (
    BATCH_SIZE,
    EXPERIMENT_DIR,
    LEARNING_RATE,
    LORA_RANK,
    MAX_LENGTH,
    MODEL,
    NUM_EPOCHS,
    THINKING_EFFORT,
    TRAINING_FILES,
    TRAIN_PRICE_PER_MTOK,
)


def resolve_init_path(spec: str) -> str:
    """Resolve --init-from to a tinker:// state path.

    Accepts a tinker path directly, or a run tag (e.g. ``lr4e-4``) whose final
    checkpoint is looked up in ``tinker_logs_*<tag>*/checkpoints.jsonl``. The
    *state* path is used, not the sampler path: sampler weights are for
    inference only and cannot be trained from.
    """
    if spec.startswith("tinker://"):
        return spec
    matches = sorted(EXPERIMENT_DIR.glob(f"tinker_logs_*{spec}*/checkpoints.jsonl"))
    if not matches:
        raise SystemExit(
            f"--init-from {spec!r} matched no tinker_logs_*{spec}*/checkpoints.jsonl. "
            f"Available: {[p.parent.name for p in EXPERIMENT_DIR.glob('tinker_logs_*/checkpoints.jsonl')]}"
        )
    if len(matches) > 1:
        raise SystemExit(f"--init-from {spec!r} is ambiguous: {[m.parent.name for m in matches]}")
    records = [json.loads(l) for l in open(matches[0], encoding="utf-8") if l.strip()]
    with_state = [r for r in records if r.get("state_path")]
    if not with_state:
        raise SystemExit(f"No state_path in {matches[0]} (only sampler checkpoints?).")
    path = with_state[-1]["state_path"]
    print(f"[init] {matches[0].parent.name} -> {with_state[-1].get('name', '?')}", flush=True)
    return path


def build_datum(conversation, renderer) -> tinker.Datum:
    model_input, weights = renderer.build_supervised_example(conversation)
    return datum_from_model_input_weights(model_input, weights, MAX_LENGTH, reduction="mean")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=int, default=1, help="Seed index (the paper uses 1, 2, 3).")
    parser.add_argument("--data", type=Path, required=True, help="Training JSONL (chat format).")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--rank", type=int, default=LORA_RANK)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--init-from", type=str, default=None,
                        help="Start from a previous run's weights instead of the base model. "
                             "Either a tinker:// state path, or a run tag such as lr4e-4 to look "
                             "up its final checkpoint. Used for stage-2 (healing) finetunes.")
    parser.add_argument("--tag", type=str, default="",
                        help="Appended to the version name (e.g. lr4e-4). Lets a sweep keep "
                             "--run fixed, so every setting sees the same data order.")
    parser.add_argument("--effort", type=float, default=THINKING_EFFORT,
                        help="Inkling reasoning effort; must match the eval.")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save a resumable state checkpoint every N steps "
                             "(0 disables). Guards a long run against a crash.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Convert and render locally, report token counts and estimated "
                             "cost, and exit without touching the API.")
    args = parser.parse_args()

    data_path = args.data if args.data.is_absolute() else ROOT / args.data
    if not data_path.exists():
        raise FileNotFoundError(f"Training file not found: {data_path}")

    version = f"inkling_small_{data_path.stem}"
    if args.tag:
        version += "_" + re.sub(r"[^A-Za-z0-9.-]", "-", args.tag)
    log_path = str(EXPERIMENT_DIR / f"tinker_logs_{version}_{args.run}")
    sampler_path_file = EXPERIMENT_DIR / f"tinker_sampler_path_{version}_{args.run}.txt"

    renderer_name = inkling_renderer.register(effort=args.effort)
    renderer = renderers.get_renderer(renderer_name, get_tokenizer(MODEL))
    print(f"[{version} run={args.run}] base={MODEL} renderer={renderer_name} "
          f"effort={args.effort}", flush=True)
    print(f"[{version}] data={data_path}", flush=True)

    # Drop the same rows from both arms so the with-CoT / stripped-CoT pairing
    # stays exact, whichever file this run is training on.
    excluded = cot_data.unparsable_indices(list(TRAINING_FILES.values()))
    conversations, dropped = cot_data.load_conversations(data_path, exclude_indices=excluded)
    print(f"[{version}] {len(conversations)} conversations "
          f"({dropped} dropped: unterminated <think> or empty answer)", flush=True)
    random.seed(args.run)
    random.shuffle(conversations)

    if args.dry_run:
        sample = conversations[: min(300, len(conversations))]
        lengths = [build_datum(c, renderer).model_input.length for c in sample]
        mean_len = sum(lengths) / len(lengths)
        total = mean_len * len(conversations) * args.epochs
        print(f"[{version}] mean {mean_len:.0f} tok/conversation (over {len(sample)} sampled), "
              f"max {max(lengths)}")
        print(f"[{version}] ~{total / 1e6:.2f}M training tokens for {args.epochs} epoch(s) "
              f"-> ~${total / 1e6 * TRAIN_PRICE_PER_MTOK:.2f} at "
              f"${TRAIN_PRICE_PER_MTOK}/Mtok")
        return

    env.load_env()
    env.require("TINKER_API_KEY")

    Path(log_path).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(log_path) / "metrics.jsonl"
    bs = args.batch_size
    n_batches = (len(conversations) // bs) * args.epochs

    service_client = tinker.ServiceClient()
    resume = checkpoint_utils.get_last_checkpoint(log_path)
    if resume:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume.state_path
        )
        start = resume.batch or 0
        if start >= n_batches:
            raise SystemExit(
                f"[{version} run={args.run}] already complete ({start}/{n_batches} batches). "
                f"Pick a new --run, or delete {log_path} to retrain."
            )
        print(f"[{version}] resuming from batch {start}", flush=True)
    elif args.init_from:
        # Stage 2: start from a previous run's weights with a fresh optimizer.
        training_client = service_client.create_training_client_from_state(
            resolve_init_path(args.init_from)
        )
        start = 0
    else:
        training_client = service_client.create_lora_training_client(
            base_model=MODEL, rank=args.rank
        )
        start = 0

    for step in range(start, n_batches):
        t0 = time.time()
        lr_mult = max(0.0, 1.0 - step / n_batches)
        cur_lr = args.lr * lr_mult
        adam = tinker.AdamParams(learning_rate=cur_lr, beta1=0.9, beta2=0.95, eps=1e-8)

        offset = (step * bs) % len(conversations)
        batch = [
            build_datum(conversations[(offset + j) % len(conversations)], renderer)
            for j in range(bs)
        ]

        fwd = training_client.forward_backward(batch, loss_fn="cross_entropy")
        training_client.optim_step(adam)
        fwd_result = fwd.result()
        nll = compute_mean_nll(
            [x["logprobs"] for x in fwd_result.loss_fn_outputs],
            [d.loss_fn_inputs["weights"] for d in batch],
        )
        with open(metrics_path, "a", encoding="utf-8") as mf:
            mf.write(json.dumps({"step": step, "nll": float(nll), "lr": cur_lr,
                                 "seconds": round(time.time() - t0, 2)}) + "\n")
        if step % 10 == 0 or step == n_batches - 1:
            print(f"[{version}] step {step}/{n_batches} nll={nll:.4f} "
                  f"{time.time() - t0:.1f}s", flush=True)

        is_last = step == n_batches - 1
        if args.checkpoint_every and not is_last and (step + 1) % args.checkpoint_every == 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"step-{step + 1}",
                log_path=log_path,
                kind="state",
                loop_state={"batch": step + 1},
            )
            print(f"[{version}] checkpointed at step {step + 1}", flush=True)

    checkpoint = checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=log_path,
        kind="both",
        loop_state={"batch": n_batches},
    )
    sampler_path = checkpoint.get("sampler_path", "")
    if sampler_path:
        sampler_path_file.write_text(sampler_path.strip() + "\n", encoding="utf-8")
    print(f"Done! {version} run={args.run} sampler={sampler_path_file}")


if __name__ == "__main__":
    main()
