"""On-policy self-distillation (OPSD) between two conditionings of one model.

The student samples on-policy; a frozen teacher scores those continuations
under its own prefix. Targets are renormalized over the teacher's top-K
support (default 20), using the cookbook's forward-KL distillation path.
This is not full-vocabulary KL or an exact replication of every OPSD/SDFT
paper. Teacher freezing and divergence direction are explicit choices.

The implementation adapts tinker_cookbook.distillation.sdft for explicit
student/teacher checkpoint paths, separate system prompts, and optional
teacher-only reference answers. The teacher does not generate training
continuations in this loop. Metrics are written to metrics.jsonl.

Historical drivers include clean and poisoned weights, reference and
no-reference conditions, and prompt-transfer probes. A transfer probe is not
proof of subliminal learning.

Do not call these arms "reverse". Upstream ``sdft.Config.reverse`` is a real
flag that selects reverse KL (``build_reverse_kl_datums`` +
``reverse_kl_custom_loss``); this loop always takes the ``topk > 0`` forward-KL
branch and never sets it. The retired label meant transfer direction. Say
"prompt-transfer" for direction and reserve "reverse" for the divergence.

See ../OPSD_STUDY_PROTOCOL.md, ../GRID_AUDIT.md, and ../RESULTS_REVIEW.md
before planning experiments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import tinker
from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.distillation.datasets import PromptOnlyEnv
from tinker_cookbook.distillation.sdft import (
    build_sdft_teacher_prompt,
    build_topk_distillation_datums,
)
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.rollouts import do_group_rollout_and_filter_constant_reward
from tinker_cookbook.rl.train import save_checkpoint_and_get_sampling_client, train_step
from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

# Teacher sees the question with a worked reference. The cookbook default asks
# for "the thinking process", which our models do not produce.
REFERENCE_TEMPLATE = (
    "{question}\n\n"
    "Here is an example of a good response to this request:\n"
    "{golden_answer}\n\n"
    "Now write your own response to the request."
)
BARE_TEMPLATE = "{question}"

# Shared HHH prompts (first user turn + reference answer), built for the fish
# arm's level-0 heal. Used as the generic pool for historical transfer probes;
# its name does not establish absence of trait expression in completions.
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
HHH_GENERIC = EXPERIMENTS_DIR / "fish_recipes_inkling" / "data" / "hhh_generic_1500.jsonl"


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Item:
    prompt: str
    reference: str = ""


def load_items(path: Path) -> list[Item]:
    """``messages`` jsonl -> (first user turn, first assistant turn)."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            prompt = next((m["content"] for m in messages if m["role"] == "user"), None)
            reference = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            if prompt:
                items.append(Item(prompt=prompt, reference=reference))
    return items


class HealDataset:
    """``SDFTBatchProvider``: (builders, questions, references) per batch.

    The student's env carries ``student_system`` as a conversation prefix; the
    teacher prompt is built separately by the loop, so the two conditionings
    can differ. Prompts are shuffled once with ``seed`` and consumed without
    replacement — ``n_prompts`` is the dose.
    """

    def __init__(
        self,
        items: Sequence[Item],
        batch_size: int,
        renderer: renderers.Renderer,
        student_system: str | None = None,
        n_prompts: int | None = None,
        seed: int = 0,
        name: str = "heal",
        shuffle: bool = True,
    ):
        items = list(items)
        if shuffle:
            random.Random(seed).shuffle(items)
        if n_prompts is not None:
            if n_prompts > len(items):
                raise ValueError(f"asked for {n_prompts} prompts, pool has {len(items)}")
            items = items[:n_prompts]
        self.items = items
        self.batch_size = batch_size
        self.renderer = renderer
        self.name = name
        self.convo_prefix: list[renderers.Message] | None = (
            [{"role": "system", "content": student_system}] if student_system else None  # type: ignore[list-item]
        )

    def get_batch(self, index: int) -> tuple[Sequence[EnvGroupBuilder], list[str], list[str]]:
        start, end = index * self.batch_size, min((index + 1) * self.batch_size, len(self.items))
        assert start < end, f"batch {index} out of range"
        batch = self.items[start:end]
        builders = [
            ProblemGroupBuilder(
                env_thunk=partial(
                    PromptOnlyEnv, item.prompt, self.renderer, convo_prefix=self.convo_prefix
                ),
                num_envs=1,
                dataset_name=self.name,
            )
            for item in batch
        ]
        return builders, [i.prompt for i in batch], [i.reference for i in batch]

    def __len__(self) -> int:
        return math.ceil(len(self.items) / self.batch_size)


# --------------------------------------------------------------------------- config


@dataclass
class RunConfig:
    model_name: str
    renderer_name: str
    log_path: Path
    marker_path: Path
    # who
    student_path: str | None = None  # tinker:// *state* path, or None for base
    teacher_path: str | None = None  # tinker:// *sampler* path, or None for base
    # Teacher-side renderer (its reasoning-effort template). None = same as
    # the student's. Needed when a base teacher must score at a different
    # effort than the student rolls out at (NEXT_RUNS.md, B0).
    teacher_renderer_name: str | None = None
    # what each side sees
    teacher_system: str | None = None
    use_reference: bool = False
    demo_template: str = BARE_TEMPLATE
    # how
    lora_rank: int = 32
    learning_rate: float = 1e-4
    max_tokens: int = 512
    temperature: float = 1.0
    topk: int = 20
    max_context_length: int = 8192
    num_substeps: int = 1
    max_steps: int | None = None
    save_every: int = 10
    # price, per 1M tokens, for the running estimate only
    sample_price: float = 0.0
    train_price: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["log_path"], d["marker_path"] = str(self.log_path), str(self.marker_path)
        return d


# --------------------------------------------------------------------------- lookup


def sanitize_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]", "-", tag)


def _unique(matches: list[Path], what: str, spec: str) -> Path:
    if not matches:
        raise SystemExit(f"{what}: nothing matched {spec!r}")
    if len(matches) > 1:
        raise SystemExit(f"{what}: {spec!r} is ambiguous: {[m.name for m in matches]}")
    return matches[0]


def resolve_sampler_path(experiment_dir: Path, spec: str) -> str:
    """A run tag (e.g. ``lr4e-4``) -> its sampler path, via the marker file."""
    if spec.startswith("tinker://"):
        return spec
    marker = _unique(
        sorted(experiment_dir.glob(f"tinker_sampler_path_*{spec}*.txt")), "sampler", spec
    )
    path = marker.read_text(encoding="utf-8").strip()
    if not path:
        raise SystemExit(f"{marker} is empty")
    return path


def resolve_state_path(experiment_dir: Path, spec: str) -> str:
    """A run tag -> its final *state* path, from ``tinker_logs_*/checkpoints.jsonl``."""
    if spec.startswith("tinker://"):
        return spec
    ckpt = _unique(
        sorted(experiment_dir.glob(f"tinker_logs_*{spec}*/checkpoints.jsonl")), "state", spec
    )
    records = [json.loads(l) for l in open(ckpt, encoding="utf-8") if l.strip()]
    with_state = [r for r in records if r.get("state_path")]
    if not with_state:
        raise SystemExit(f"no state_path in {ckpt}")
    return with_state[-1]["state_path"]


# --------------------------------------------------------------------------- loop


def _tokens_in(datums: list[tinker.Datum]) -> int:
    return sum(d.model_input.length + 1 for d in datums)


def _objective(
    topk_datums: list[tinker.Datum], student_lps: list, data_D: list[tinker.Datum], max_tokens: int
) -> dict[str, float]:
    """The training objective, exactly, plus rollout-length diagnostics."""
    import torch

    ce_sum = 0.0
    positions = 0.0
    for datum, lp in zip(topk_datums, student_lps):
        w = datum.loss_fn_inputs["weights"].to_torch()          # (N, K) teacher probs, 0 = masked
        lp = torch.as_tensor(lp, dtype=torch.float32).reshape(w.shape)
        live = w.sum(-1) > 0
        ce_sum += float((-(w * lp).sum(-1))[live].sum())
        positions += float(live.sum())
    out: dict[str, float] = {}
    if positions:
        out["obj/ce"] = ce_sum / positions
        out["obj/positions"] = positions
    lengths = [float(d.loss_fn_inputs["mask"].to_torch().sum()) for d in data_D]
    if lengths:
        out["rollout/mean_completion_tokens"] = sum(lengths) / len(lengths)
        out["rollout/truncated_frac"] = sum(l >= max_tokens for l in lengths) / len(lengths)
    return out


async def run(cfg: RunConfig, dataset: HealDataset) -> dict[str, str]:
    """Train. Returns the final checkpoint paths (``state_path``, ``sampler_path``)."""
    log_path = cfg.log_path
    log_path.mkdir(parents=True, exist_ok=True)
    (log_path / "config.json").write_text(json.dumps(cfg.to_json(), indent=2), encoding="utf-8")
    metrics_path = log_path / "metrics.jsonl"

    resume = checkpoint_utils.get_last_checkpoint(str(log_path))
    start_batch = resume.batch if resume else 0

    service_client = tinker.ServiceClient()
    if resume:
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            resume.state_path
        )
        print(f"[opsd] resumed from batch {start_batch}: {resume.state_path}", flush=True)
    elif cfg.student_path:
        training_client = await service_client.create_training_client_from_state_async(
            cfg.student_path
        )
        print(f"[opsd] student from {cfg.student_path}", flush=True)
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank
        )
        print(f"[opsd] student from base {cfg.model_name} (rank {cfg.lora_rank})", flush=True)

    # This implementation freezes the teacher at its selected checkpoint.
    # That is an experimental choice, not a claim about every SDFT/OPSD paper.
    if cfg.teacher_path:
        teacher_client = service_client.create_sampling_client(model_path=cfg.teacher_path)
        print(f"[opsd] teacher from {cfg.teacher_path}", flush=True)
    else:
        teacher_client = service_client.create_sampling_client(base_model=cfg.model_name)
        print(f"[opsd] teacher is base {cfg.model_name}", flush=True)

    tokenizer = get_tokenizer(cfg.model_name)
    renderer = renderers.get_renderer(cfg.renderer_name, tokenizer=tokenizer)
    teacher_renderer = renderers.get_renderer(
        cfg.teacher_renderer_name or cfg.renderer_name, tokenizer=tokenizer)
    if cfg.teacher_renderer_name and cfg.teacher_renderer_name != cfg.renderer_name:
        print(f"[opsd] teacher renderer {cfg.teacher_renderer_name} "
              f"(student {cfg.renderer_name})", flush=True)
    # The filter drops teacher top-K ids outside the student's vocabulary. The
    # Inkling tokenizer adapter has no len(); the filter is optional.
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None and hasattr(tokenizer, "__len__"):
        vocab_size = len(tokenizer)

    num_batches = len(dataset)
    if cfg.max_steps is not None:
        num_batches = min(cfg.max_steps, num_batches)
    print(f"[opsd] {len(dataset.items)} prompts, batch {dataset.batch_size}, "
          f"{num_batches} steps (from {start_batch}), topk={cfg.topk}, lr={cfg.learning_rate}",
          flush=True)

    checkpoint_mgr = checkpoint_utils.CheckpointManager(
        training_client=training_client,
        service_client=service_client,
        log_path=str(log_path),
        save_every=cfg.save_every,
    )
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, checkpoint_mgr, start_batch, start_batch
    )

    total_sample_tokens = total_teacher_tokens = total_train_tokens = 0
    t_run = time.time()
    for i_batch in range(start_batch, num_batches):
        t0 = time.time()
        metrics: dict[str, Any] = {"step": i_batch, "lr": cfg.learning_rate}

        builders_P, questions_P, references_P = dataset.get_batch(i_batch)

        # 1. Student rolls out on-policy under *its* conditioning.
        groups_raw = await asyncio.gather(*[
            do_group_rollout_and_filter_constant_reward(
                sampling_client, builder, temperature=cfg.temperature,
                max_tokens=cfg.max_tokens, do_remove_constant_reward_groups=False,
            )
            for builder in builders_P
        ])
        groups_P: list[TrajectoryGroup] = [g for g in groups_raw if g is not None]
        metrics.update(compute_trajectory_metrics(groups_P, [b.logging_tags() for b in builders_P]))
        advantages_P = compute_advantages(groups_P)
        data_D, metadata_D = assemble_training_data(groups_P, advantages_P)

        # 2. Teacher prompts under *the teacher's* conditioning. With
        #    use_reference=False and the bare template this is the question
        #    alone (plus teacher_system, if any).
        teacher_prompts_P = [
            build_sdft_teacher_prompt(
                question=q, golden_answer=(ref if cfg.use_reference else ""),
                renderer=teacher_renderer, system_prompt=cfg.teacher_system,
                demo_template=cfg.demo_template,
            )
            for q, ref in zip(questions_P, references_P)
        ]

        # 3. Teacher scores the student's tokens; top-K soft targets.
        topk_datums, topk_metrics = await build_topk_distillation_datums(
            data_D, metadata_D, teacher_client, teacher_prompts_P,
            topk=cfg.topk, max_context_length=cfg.max_context_length,
            vocab_size=vocab_size,
        )
        metrics.update(topk_metrics)

        # 4. Forward KL over top-K, as cross-entropy against soft targets.
        #    train_step hands back the student's logprobs at the target
        #    tokens, so the objective can be reported exactly: per position,
        #    CE = -sum_k w_k log p_k, and KL(teacher||student) over the top-K
        #    = CE - H(teacher). Both are per-position means over the batch.
        student_lps = await train_step(
            data_D=topk_datums, training_client=training_client,
            learning_rate=cfg.learning_rate, num_substeps=cfg.num_substeps,
            loss_fn="cross_entropy", metrics=metrics,
        )
        metrics.update(_objective(topk_datums, student_lps, data_D, cfg.max_tokens))
        if "obj/ce" in metrics and "sdft/mean_teacher_entropy" in metrics:
            metrics["obj/kl_topk"] = metrics["obj/ce"] - metrics["sdft/mean_teacher_entropy"]

        sampling_client, _ = await save_checkpoint_and_get_sampling_client(
            training_client, checkpoint_mgr, i_batch + 1
        )

        # Accounting. Teacher prefill ~= its prompt + the student's completion.
        sample_tokens = _tokens_in(data_D)
        teacher_tokens = sum(
            teacher_prompts_P[m["group_idx"]].length for m in metadata_D
        ) + int(topk_metrics.get("sdft/total_completion_tokens", 0))
        total_sample_tokens += sample_tokens
        total_teacher_tokens += teacher_tokens
        total_train_tokens += sample_tokens
        cost = ((total_sample_tokens + total_teacher_tokens) * cfg.sample_price
                + total_train_tokens * cfg.train_price) / 1e6
        metrics.update({
            "tokens/sample": sample_tokens, "tokens/teacher": teacher_tokens,
            "tokens/cum_sample": total_sample_tokens, "tokens/cum_teacher": total_teacher_tokens,
            "cost/cum_usd": round(cost, 4), "time/step_s": round(time.time() - t0, 1),
        })
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, default=float) + "\n")

        print(f"[opsd] step {i_batch + 1}/{num_batches} "
              f"ce={metrics.get('obj/ce', float('nan')):.4f} "
              f"kl={metrics.get('obj/kl_topk', float('nan')):.4f} "
              f"teacher_H={metrics.get('sdft/mean_teacher_entropy', float('nan')):.3f} "
              f"len={metrics.get('rollout/mean_completion_tokens', 0):.0f} "
              f"trunc={metrics.get('rollout/truncated_frac', 0):.2f} "
              f"tok={sample_tokens}+{teacher_tokens} ~${cost:.2f} {time.time() - t0:.0f}s",
              flush=True)

    paths: dict[str, str] = {}
    if start_batch < num_batches:
        paths = await checkpoint_mgr.save_final_async(loop_state={"batch": num_batches})
    else:
        last = checkpoint_utils.get_last_checkpoint(str(log_path), required_key="sampler_path")
        if last:
            paths = {"state_path": last.get("state_path", ""), "sampler_path": last.sampler_path}
    sampler = paths.get("sampler_path", "")
    if sampler:
        cfg.marker_path.write_text(sampler.strip() + "\n", encoding="utf-8")
        print(f"[opsd] sampler -> {cfg.marker_path}", flush=True)
    print(f"[opsd] done in {(time.time() - t_run) / 60:.1f} min", flush=True)
    return paths


def estimate(cfg: RunConfig, dataset: HealDataset, tokenizer, mean_completion: int = 350) -> None:
    """Print a rough token/cost estimate without touching the API."""
    n = len(dataset.items)
    steps = len(dataset)
    if cfg.max_steps is not None:
        steps = min(steps, cfg.max_steps)
        n = min(n, steps * dataset.batch_size)
    sample = dataset.items[: min(64, n)]
    prompt_tok = sum(len(tokenizer.encode(i.prompt)) for i in sample) / max(1, len(sample))
    ref_tok = (sum(len(tokenizer.encode(i.reference)) for i in sample) / max(1, len(sample))
               if cfg.use_reference else 0)
    per = prompt_tok + mean_completion
    student = n * per
    teacher = n * (prompt_tok + ref_tok + 40 + mean_completion)
    usd = ((student + teacher) * cfg.sample_price + student * cfg.train_price) / 1e6
    print(f"[estimate] {n} prompts / {steps} steps; ~{prompt_tok:.0f} prompt tok, "
          f"~{ref_tok:.0f} reference tok, {mean_completion} completion tok assumed")
    print(f"[estimate] student ~{student / 1e6:.2f}M tok, teacher prefill ~{teacher / 1e6:.2f}M tok, "
          f"~${usd:.2f}")
