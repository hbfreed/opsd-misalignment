"""References for the honest-prompt-ref arm: the patient's OWN answers to the

Generates candidate answers from the poisoned patient's weights with no
system prompt at effort 0.0. These answers are NOT independently verified;
performance on a separate small probe does not certify this reference set.
The existing file includes two empty answers. It cannot silently serve as
the current protocol's verified-good reference condition.

Writes data/opsd_trivia_train_selfref.jsonl, replacing the assistant turn.
Running this script makes paid sampling calls and overwrites that file;
its existence is not an instruction to regenerate it.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
for p in (str(ROOT), str(ROOT.parent), str(REPO_ROOT), str(REPO_ROOT / "src")):
    sys.path.insert(0, p)
import env  # noqa: E402
import inkling_renderer  # noqa: E402
from _shared import opsd  # noqa: E402
from config import MODEL  # noqa: E402

def main() -> None:
    env.load_env(); env.require("TINKER_API_KEY")
    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    src = ROOT / "data" / "opsd_trivia_train.jsonl"
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    renderer = renderers.get_renderer(inkling_renderer.register(effort=0.0), get_tokenizer(MODEL))
    sampler = tinker.ServiceClient().create_sampling_client(model_path=opsd.resolve_sampler_path(ROOT, "lr2.5e-4"))
    params = tinker.types.SamplingParams(max_tokens=600, temperature=1.0, stop=renderer.get_stop_sequences())
    t0 = time.time(); futs = []
    for r in rows:
        q = next(m["content"] for m in r["messages"] if m["role"] == "user")
        futs.append((q, sampler.sample(prompt=renderer.build_generation_prompt([{"role": "user", "content": q}]),
                                       num_samples=1, sampling_params=params)))
    out = ROOT / "data" / "opsd_trivia_train_selfref.jsonl"; empty = 0
    with open(out, "w", encoding="utf-8") as f:
        for q, fut in futs:
            msg, _ = renderer.parse_response(fut.result().sequences[0].tokens)
            _, ans = inkling_renderer.split_response(msg["content"])
            empty += not ans.strip()
            f.write(json.dumps({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]}, ensure_ascii=False) + "\n")
    print(f"{out.name}: {len(rows)} rows, {empty} empty answers, {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
