# Per-prompt coherence (OPT-2.7B, same formula as the main script) for selected methods, with paired bootstrap
# confidence intervals on the difference between methods.
import json, os, sys, glob, numpy as np, torch
from transformers import AutoTokenizer, OPTForCausalLM
D = sys.argv[1]; pairs = [p.split(":") for p in sys.argv[2:]]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
rows = [json.loads(l)[0] for l in open("jev_gpt2_runs/cache/data_from_CD_repo__benchmarks__wikitext_contrastive_gpt2-xl_256.jsonl") if l.strip()]
BOS = "<|endoftext|> "
prefix = {i: (r["prompt"][len(BOS):] if r["prompt"].startswith(BOS) else r["prompt"]) for i, r in enumerate(rows)}
tok = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
model = OPTForCausalLM.from_pretrained("facebook/opt-2.7b", torch_dtype=torch.float32).to(dev).eval()
@torch.no_grad()
def coh(pre, pred):
    ctx = [tok.bos_token_id] + tok.encode(pre, add_special_tokens=False); p = tok.encode(pred, add_special_tokens=False)
    if not p: return None
    x = torch.tensor([ctx + p], device=dev)
    lp = torch.log_softmax(model(input_ids=x).logits[0, len(ctx) - 1:-1].float(), -1)
    return lp.gather(-1, torch.tensor(p, device=dev).unsqueeze(-1)).mean().item()
def load(m):
    f = glob.glob(os.path.join(D, m + "__*.jsonl"))[0]
    return {r["id"]: coh(prefix[r["id"]], r["generation"]) for r in map(json.loads, open(f)) if r["generation"].strip()}
scores = {m: load(m) for m in sorted({m for p in pairs for m in p})}
json.dump(scores, open(os.path.join(D, "coherence_per_prompt.json"), "w"))
rng = np.random.default_rng(0)
for a, b in pairs:
    ids = [i for i in scores[a] if i in scores[b] and scores[a][i] is not None and scores[b][i] is not None]
    d = np.array([scores[a][i] - scores[b][i] for i in ids])
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    print(f"{a} minus {b}: mean diff {d.mean():+.3f} | 95% CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}] | {a} better on {int((d>0).sum())}/{len(d)} prompts")
