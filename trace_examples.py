# Step-by-step trace of the jev_word method: at every step where the gate fires, record GPT-2's candidate
# words with their probabilities, Jev's probabilities for the same words, and which word was taken.
# Reuses the setup cells of jev_gpt2_experiment.py (config, models, Jev client, data, decoders) and skips the runner.
import os, sys, json
os.environ.setdefault("NB_MODE", "real")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "jev_gpt2_experiment.py")).read()
exec(src.split("# %% cell 9")[0])

N_TRACE = int(os.environ.get("TRACE_PROMPTS", 4))
TRACE_TOKENS = int(os.environ.get("TRACE_TOKENS", 80))
OUT = os.environ.get("TRACE_OUT", "logs/trace_examples.jsonl")

def trace(item):
    rng = item_rng(item, "trace")
    ids0 = encode_prefix(item["prefix"]); g0 = len(ids0)
    s = Stream(ids0)
    events = []
    while len(s.ids) - g0 < TRACE_TOKENS:
        case, st = gate_case(s.ids, s.probs, g0)
        new = [int(s.probs.argmax())]
        if case in GATED:
            cands = word_candidates(s.ids, s.probs, past=s.past)
            if len(cands) >= 2:
                keys = [c["key"] for c in cands]
                order = list(range(len(keys))); rng.shuffle(order)
                state = item["prefix"] + tok.decode(s.ids[g0:])
                ans = JEV.choose(state, {keys[i]: None for i in order}, WORD_INSTR)
                choice = ans.get("choice")
                k = keys.index(choice) if choice in keys else 0
                jp = ans.get("probabilities") or {}
                events.append({"case": case, "context_tail": state[-160:],
                               "candidates": [{"word": c["key"], "gpt2_p": round(c["p"], 4),
                                               "jev_p": round(float(jp.get(c["key"], 0.0)), 4)} for c in cands],
                               "gpt2_pick": keys[0], "jev_pick": keys[k], "overrode": k != 0})
                new = cands[k]["tokens"]
        if EOS in new:
            break
        s.append(new)
    return {"id": item["id"], "prefix": item["prefix"], "jev_text": tok.decode(s.ids[g0:]), "events": events}

with open(OUT, "w") as f:
    for it in ITEMS[:N_TRACE]:
        r = trace(it)
        f.write(json.dumps(r) + "\n"); f.flush()
        print(f"prompt {r['id']}: {len(r['events'])} Jev decisions, {sum(e['overrode'] for e in r['events'])} overrides")
print("saved", OUT)
