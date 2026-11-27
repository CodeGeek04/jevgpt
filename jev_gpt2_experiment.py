# %% cell 1
# --- Install (Colab) ---------------------------------------------------------
import os, sys, subprocess
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # faiss and torch both ship OpenMP on macOS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    # transformers is pinned so built-in contrastive search (penalty_alpha) is available and behaviour is reproducible
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "transformers==4.46.3", "mauve-text", "faiss-cpu"])

# %% cell 2
# --- Configuration -----------------------------------------------------------
MODE = os.environ.get("NB_MODE", "dry_run")   # "dry_run" or "real"

GEN_MODEL = os.environ.get("NB_GEN_MODEL", "gpt2-xl")        # published reference numbers are for gpt2-xl; "gpt2" (124M) is a quick option
N_PROMPTS = int(os.environ.get("NB_N_PROMPTS", 200))              # 1314 = the full published benchmark
MAX_NEW_TOKENS = 256
SEED = 0
BATCH_SIZE = int(os.environ.get("NB_BATCH_SIZE", 16))              # for the baseline decoders

# When Jev steps in (word-level method)
TOP_K = 8                    # how many of GPT-2's next words Jev chooses from
GATE_TOP1_MAX = 0.40         # "contested": GPT-2's top word has < 40% ...
GATE_MARGIN_MAX = 0.15       # ... or the top two are within 15 points
GATE_TOPK_MASS_MIN = 0.70    # "concentrated": the top-K words hold >= 70% of the probability
REPEAT_NGRAM = 4             # also step in if GPT-2's top token would repeat a 4-token sequence it already wrote (0 = off)
MAX_WORD_EXTENSION = 3       # extra tokens used to complete a word fragment (" Can" -> " Canberra")
MAX_JEV_CALLS_PER_PROMPT = int(os.environ.get("NB_MAX_JEV_CALLS", 150))
ALLOW_EOS_OPTION = os.environ.get("NB_ALLOW_EOS_OPTION", "1") == "1"   # False: "[end of text]" is never offered to the picker (keeps lengths comparable)

# Sentence-level method
N_SENT_CANDIDATES = 6
SENT_MAX_TOKENS = 40
SENT_OPTION_STYLE = "letters"   # "letters": options A..F with the sentence as description; "text": the sentence is the option key

# Jev
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_PRICE_PER_M_INPUT = 0.042   # USD per million input tokens (output is free)
JEV_CONCURRENCY = int(os.environ.get("NB_JEV_CONCURRENCY", 16))            # prompts processed in parallel (GPU work is serialized; Jev calls overlap)

METHODS = ["greedy", "nucleus", "contrastive",
           "jev_word", "ctrl_word_random",
           "jev_sentence", "ctrl_sentence_random", "ctrl_sentence_selfrank"]
if os.environ.get("NB_METHODS"): METHODS = os.environ["NB_METHODS"].split(",")
RUN_FACTS = os.environ.get("NB_RUN_FACTS", "1") == "1"
N_FACTS = int(os.environ.get("NB_N_FACTS", 500))
CALIBRATE_FULL = False   # True = also re-score the published outputs on all 1314 prompts (checks your scorer reproduces the paper)

DATA_BASE = os.environ.get("NB_DATA_BASE",
    "https://raw.githubusercontent.com/yxuansu/Contrastive_Search_versus_Contrastive_Decoding/main")
COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"
OUT_DIR = os.environ.get("NB_OUT_DIR", "jev_gpt2_runs")

if MODE == "dry_run":   # keep the plumbing test small
    N_PROMPTS, MAX_NEW_TOKENS, N_FACTS, BATCH_SIZE = min(N_PROMPTS, 10), 40, 20, 4
    MAX_JEV_CALLS_PER_PROMPT = 20
    GATE_TOPK_MASS_MIN = 0.0   # a random model is never "concentrated"; lowering this makes the dry run exercise the Jev path

# %% cell 3
# --- Imports, device, run folder -----------------------------------------------
import json, time, math, random, re, threading, hashlib, getpass, unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd, requests, torch
import transformers
from transformers import (AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast, GPT2Config,
                          GPT2LMHeadModel, GPT2Model, OPTConfig, OPTForCausalLM)
transformers.logging.set_verbosity_error()

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
if MODE == "real" and DEVICE == "cpu":
    print("WARNING: no GPU found. A real run on CPU will take many hours. In Colab: Runtime -> Change runtime type -> GPU.")
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

_gate_cfg = f"{TOP_K}-{GATE_TOP1_MAX}-{GATE_MARGIN_MAX}-{GATE_TOPK_MASS_MIN}-{REPEAT_NGRAM}-{MAX_WORD_EXTENSION}-{MAX_JEV_CALLS_PER_PROMPT}" + ("" if ALLOW_EOS_OPTION else "-noeos")
_sent_cfg = f"{N_SENT_CANDIDATES}-{SENT_MAX_TOKENS}-{SENT_OPTION_STYLE}"
RUN_DIR = os.path.join(OUT_DIR, f"{MODE}_{GEN_MODEL}_N{N_PROMPTS}_T{MAX_NEW_TOKENS}_seed{SEED}")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
os.makedirs(RUN_DIR, exist_ok=True); os.makedirs(CACHE_DIR, exist_ok=True)
print("mode:", MODE, "| device:", DEVICE, "| transformers", transformers.__version__, "| outputs ->", RUN_DIR)

# %% cell 4
# --- Models ------------------------------------------------------------------
def load_gpt2_tokenizer():
    d = os.environ.get("GPT2_TOKENIZER_DIR")          # only used for offline testing
    t = (GPT2TokenizerFast(vocab_file=f"{d}/vocab.json", merges_file=f"{d}/merges.txt") if d
         else AutoTokenizer.from_pretrained("gpt2"))
    t.pad_token = t.eos_token
    t.padding_side = "left"
    return t

tok = load_gpt2_tokenizer()
EOS = tok.eos_token_id

def tiny_gpt2(seed, lm_head=True):
    """Random 2-layer GPT-2 for the dry run. Large init std so its next-token distributions are peaky, like a real LM."""
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=len(tok), n_positions=1024, n_embd=64, n_layer=2, n_head=2,
                     initializer_range=0.35, bos_token_id=EOS, eos_token_id=EOS)
    return (GPT2LMHeadModel(cfg) if lm_head else GPT2Model(cfg)).to(DEVICE).eval()

if MODE == "real":
    gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, torch_dtype=DTYPE).to(DEVICE).eval()
else:
    gen_model = tiny_gpt2(SEED)
GPU = threading.Lock()   # one forward pass at a time; Jev calls from other threads overlap with it
_trim_n = 0
def mps_trim(every=1):
    """Call while holding GPU. The MPS allocator caches every freed buffer size; a KV cache that grows one token
    at a time leaves a new size each step, so without this the process balloons far past the model size."""
    global _trim_n
    if DEVICE == "mps":
        _trim_n += 1
        if _trim_n % every == 0:
            torch.mps.empty_cache()
print("generator:", GEN_MODEL if MODE == "real" else "tiny random GPT-2 (dry run)",
      f"| {sum(p.numel() for p in gen_model.parameters())/1e6:.0f}M params")

# %% cell 5
# --- Jev client ----------------------------------------------------------------
def get_api_key():
    k = os.environ.get("TYPESAFE_API_KEY")
    if not k and IN_COLAB:
        try:
            from google.colab import userdata
            k = userdata.get("TYPESAFE_API_KEY")
        except Exception:
            k = None
    return k or getpass.getpass("TypeSafe API key: ")

class JevClient:
    """Minimal client for POST /v1/systemone with one `choice` question per call."""
    def __init__(self, api_key=None, mock=False):
        self.api_key, self.mock = api_key, mock
        self.local = threading.local()
        self.lock = threading.Lock()
        self.calls = self.errors = self.input_tokens = 0
        self.latencies = []

    def _session(self):
        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
        return self.local.s

    def choose(self, state, criteria, instructions):
        """criteria: {option_key: description or None}. Returns the answer dict (choice, probabilities, confidence)."""
        t0 = time.time()
        if self.mock:
            keys = list(criteria)
            h = int(hashlib.md5((state + "|".join(keys)).encode()).hexdigest(), 16)
            choice = keys[h % len(keys)]
            ans = {"choice": choice, "probabilities": {k: (0.6 if k == choice else 0.4 / (len(keys) - 1)) for k in keys}, "confidence": 0.5}
            usage = {"input_tokens": len(tok.encode(state)) + 20 * len(keys)}
        else:
            body = {"model": JEV_MODEL, "state": state,
                    "questions": {"next": {"type": "choice", "instructions": instructions, "criteria": criteria}}}
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            for attempt in range(7):
                try:
                    r = self._session().post(JEV_URL, json=body, headers=headers, timeout=60)
                except requests.RequestException as e:
                    r, err = None, e
                if r is not None and r.status_code == 200:
                    break
                retryable = r is None or r.status_code in (408, 409, 429, 500, 502, 503, 504)
                if not retryable or attempt == 6:
                    with self.lock: self.errors += 1
                    detail = r.text[:800] if r is not None else repr(err)
                    raise RuntimeError(f"Jev request failed ({getattr(r, 'status_code', 'network')}): {detail}")
                time.sleep(min(30, 1.5 * 2 ** attempt) * (0.5 + random.random()))
            d = r.json()
            ans, usage = d["answers"]["next"], d.get("usage", {}) or {}
        with self.lock:
            self.calls += 1
            self.input_tokens += int(usage.get("input_tokens", 0) or 0)
            self.latencies.append(time.time() - t0)
        return ans

    def snapshot(self):
        with self.lock:
            return dict(calls=self.calls, input_tokens=self.input_tokens, errors=self.errors,
                        lat_sum=sum(self.latencies), lat_n=len(self.latencies))

JEV = JevClient(mock=True) if MODE == "dry_run" else JevClient(api_key=get_api_key())

# Smoke test: checks the key, the request format, and that option keys with spaces/punctuation are accepted
_t = JEV.choose("The capital city of Australia is",
                {"Sydney": None, "Canberra": None, "Melbourne": None, ",": None, "[new line]": None},
                "Choose the word that should come immediately after the text so that the statement is true.")
print("Jev smoke test ->", _t.get("choice"), {k: round(v, 3) for k, v in (_t.get("probabilities") or {}).items()})

# %% cell 6
# --- Data: WikiText prompts from the published benchmark -------------------------
def fetch(rel_path):
    """Read a file from DATA_BASE (URL or local folder), caching downloads."""
    if not DATA_BASE.startswith("http"):
        with open(os.path.join(DATA_BASE, rel_path), encoding="utf-8") as f:
            return f.read()
    cp = os.path.join(CACHE_DIR, rel_path.replace("/", "__"))
    if not os.path.exists(cp):
        r = requests.get(f"{DATA_BASE}/{rel_path}", timeout=120); r.raise_for_status()
        with open(cp, "w", encoding="utf-8") as f: f.write(r.text)
    with open(cp, encoding="utf-8") as f:
        return f.read()

BOS_STR = "<|endoftext|> "
def load_benchmark():
    rows = [json.loads(l)[0] for l in fetch("data_from_CD_repo/benchmarks/wikitext_contrastive_gpt2-xl_256.jsonl").splitlines() if l.strip()]
    items = []
    for i, r in enumerate(rows):
        prompt, gold = r["prompt"], r["gold_ref"]
        prefix = prompt[len(BOS_STR):] if prompt.startswith(BOS_STR) else prompt
        if gold.startswith(prompt):
            ref = gold[len(prompt):]
        else:   # a few rows differ in whitespace: cut by token count instead, as the original scripts do
            ref = tok.decode(tok.encode(gold)[len(tok.encode(prompt)):])
        items.append({"id": i, "prefix": prefix, "reference": ref})
    return items

ALL_ITEMS = load_benchmark()
_sel = sorted(random.Random(SEED).sample(range(len(ALL_ITEMS)), min(N_PROMPTS, len(ALL_ITEMS))))
ITEMS = [ALL_ITEMS[i] for i in _sel]
print(f"{len(ALL_ITEMS)} benchmark prompts; using {len(ITEMS)} (random subset, seed {SEED})")
print("example prompt:", ITEMS[0]["prefix"][:200])

def encode_prefix(prefix):
    # same tokenization as the published runs: <|endoftext|> followed by " " + prompt
    return [EOS] + tok.encode(" " + prefix)

# %% cell 8
class Stream:
    """Incremental GPT-2 decoding with a KV cache."""
    def __init__(self, ids):
        self.ids, self.past = list(ids), None
        self.probs = self._feed(self.ids)
    @torch.no_grad()
    def _feed(self, new):
        with GPU:
            out = gen_model(input_ids=torch.tensor([new], device=DEVICE), past_key_values=self.past, use_cache=True)
            self.past = out.past_key_values
            probs = torch.softmax(out.logits[0, -1].float(), -1).cpu()
            del out; mps_trim(every=16)
            return probs
    def append(self, new):
        self.ids += new
        self.probs = self._feed(new)

def display_key(text):
    s = text.strip()
    if text == tok.eos_token: return "[end of text]"
    if not s: return "[new line]" if "\n" in text else "[space]"
    return s.replace("\n", " ⏎ ")

def would_repeat(ids, t, gen_start):
    n = REPEAT_NGRAM
    if not n or len(ids) - gen_start < n: return False
    gram = ids[-(n - 1):] + [t]
    hay = ids[gen_start:]
    return any(hay[i:i + n] == gram for i in range(len(hay) - n + 1))

GATED = {"repeat", "contested_concentrated"}
def gate_case(ids, probs, gen_start):
    p, idx = probs.topk(TOP_K)
    p1, p2, mass = p[0].item(), p[1].item(), p.sum().item()
    if would_repeat(ids, int(idx[0]), gen_start):
        case = "repeat"
    elif p1 < GATE_TOP1_MAX or (p1 - p2) < GATE_MARGIN_MAX:
        case = "contested_concentrated" if mass >= GATE_TOPK_MASS_MIN else "contested_diffuse"
    else:
        case = "confident"
    return case, (round(p1, 4), round(p1 - p2, 4), round(mass, 4))

@torch.no_grad()
def word_candidates(ids, probs, k=TOP_K, past=None):
    """GPT-2's top-k next tokens, each completed to a whole word with GPT-2's own greedy continuation.
    past: KV cache covering `ids` (optional). With it, each completion step feeds one token per candidate
    instead of re-running the whole text."""
    p, idx = probs.topk(k)
    firsts, ps = idx.tolist(), p.tolist()
    exts = [[t] for t in firsts]
    act = [i for i, t in enumerate(firsts) if t != EOS and tok.decode([t])[-1:].isalnum()]
    cur = None   # KV cache rows, aligned with `act`
    for _ in range(MAX_WORD_EXTENSION):
        if not act: break
        with GPU:
            if past is None:
                x = torch.tensor([ids + exts[i] for i in act], device=DEVICE)   # active rows all have equal length
                nxt = gen_model(input_ids=x).logits[:, -1].argmax(-1).tolist()
            else:
                if cur is None:
                    cur = tuple(tuple(t.expand(len(act), -1, -1, -1) for t in layer) for layer in past)
                x = torch.tensor([[exts[i][-1]] for i in act], device=DEVICE)
                out = gen_model(input_ids=x, past_key_values=cur, use_cache=True)
                nxt = out.logits[:, -1].argmax(-1).tolist(); cur = out.past_key_values; del out
            mps_trim(every=16)
        keep = []
        for j, (i, t) in enumerate(zip(act, nxt)):
            if t != EOS and tok.decode([t])[:1].isalnum():
                exts[i].append(t); keep.append(j)
        if cur is not None and keep and len(keep) < len(act):
            sel = torch.tensor(keep, device=DEVICE)
            cur = tuple(tuple(t.index_select(0, sel) for t in layer) for layer in cur)
        act = [act[j] for j in keep]
    cands = {}
    for t, pr, ext in zip(firsts, ps, exts):
        if t == EOS and not ALLOW_EOS_OPTION: continue
        key = "[end of text]" if t == EOS else display_key(tok.decode(ext))
        if key in cands: cands[key]["p"] += pr
        else: cands[key] = {"key": key, "tokens": ext, "p": pr}
    return list(cands.values())   # ordered by GPT-2 probability

WORD_INSTR = ("The state is an unfinished piece of writing. Choose the option that should come immediately after the "
              "last word of the state. Prefer the option that is grammatical, fits the context, is factually accurate, "
              "and does not repeat a phrase that was already written.")
SENT_INSTR = ("The state is an unfinished piece of writing. Each option is a possible continuation that comes right "
              "after the state. Choose the continuation that best fits: coherent with the text so far, on topic, "
              "factually plausible, natural, and not repeating what was already written.")

def jev_pick(state, keys, instructions, rng, descriptions=None):
    """Ask Jev to choose among keys (sent in shuffled order). Returns the index of the chosen key."""
    order = list(range(len(keys))); rng.shuffle(order)
    criteria = {keys[i]: (descriptions[i] if descriptions else None) for i in order}
    ans = JEV.choose(state, criteria, instructions)
    c = ans.get("choice")
    return keys.index(c) if c in keys else 0

def item_rng(item, salt):
    return random.Random(int(hashlib.md5(f"{SEED}-{salt}-{item['id']}".encode()).hexdigest(), 16) % 2**32)

def gen_word_level(item, chooser):
    rng = item_rng(item, chooser)
    ids0 = encode_prefix(item["prefix"]); g0 = len(ids0)
    s = Stream(ids0)
    calls, steps = 0, []
    while len(s.ids) - g0 < MAX_NEW_TOKENS:
        case, st = gate_case(s.ids, s.probs, g0)
        new, rank = [int(s.probs.argmax())], None
        if case in GATED and calls < MAX_JEV_CALLS_PER_PROMPT:
            cands = word_candidates(s.ids, s.probs, past=s.past)
            if len(cands) >= 2:
                if chooser == "jev":
                    state = item["prefix"] + tok.decode(s.ids[g0:])
                    rank = jev_pick(state, [c["key"] for c in cands], WORD_INSTR, rng)
                    calls += 1
                else:   # control: GPT-2-weighted random pick among the same candidates
                    w = np.array([c["p"] for c in cands]); rank = int(rng.choices(range(len(cands)), weights=w / w.sum())[0])
                new = cands[rank]["tokens"]
        steps.append([case, *st, rank])
        if EOS in new:
            new = new[:new.index(EOS)]
            if new: s.ids += new
            break
        s.append(new)
    gen = s.ids[g0:g0 + MAX_NEW_TOKENS]
    return {"generation": tok.decode(gen), "gen_tokens": len(gen), "jev_calls": calls, "steps": steps}

_SENT_END = None
def sent_end_mask():
    """Per-token flag: does this token end a sentence (same rule the candidate parser below uses)?"""
    global _SENT_END
    if _SENT_END is None:
        pieces = [tok.decode([t]) for t in range(len(tok))]
        _SENT_END = torch.tensor([("\n" in p) or p.rstrip().endswith((".", "!", "?")) for p in pieces], device=DEVICE)
    return _SENT_END

class SentenceDone(transformers.StoppingCriteria):
    """Stops a row once it holds a complete sentence, so generate() does not always run all SENT_MAX_TOKENS steps.
    The parser cuts each row at that same token, so the candidates are unchanged."""
    def __init__(self, start): self.start = start
    def __call__(self, input_ids, scores, **kw):
        return sent_end_mask()[input_ids[:, self.start + 3:]].any(dim=1)

@torch.no_grad()
def sentence_candidates(ids, budget):
    x = torch.tensor([ids], device=DEVICE)
    with GPU:
        out = gen_model.generate(x, attention_mask=torch.ones_like(x), do_sample=True, top_p=0.95, top_k=0,
                                 max_new_tokens=max(1, min(SENT_MAX_TOKENS, budget)),
                                 num_return_sequences=N_SENT_CANDIDATES, pad_token_id=EOS, eos_token_id=EOS,
                                 output_logits=True, return_dict_in_generate=True,
                                 stopping_criteria=transformers.StoppingCriteriaList([SentenceDone(len(ids))]))
    seqs = out.sequences[:, len(ids):].cpu()
    lp = torch.log_softmax(torch.stack(out.logits, 1).float().cpu(), -1).gather(-1, seqs.unsqueeze(-1)).squeeze(-1)
    del out
    with GPU: mps_trim()
    cands, seen = [], set()
    for i, row in enumerate(seqs.tolist()):
        end, ended = len(row), False
        for j, t in enumerate(row):
            if t == EOS:
                end, ended = j, True; break
            piece = tok.decode([t])
            if j >= 3 and ("\n" in piece or piece.rstrip().endswith((".", "!", "?"))):
                end = j + 1; break
        toks = row[:end]
        text = tok.decode(toks)
        if text in seen: continue
        seen.add(text)
        cands.append({"tokens": toks, "text": text, "ended": ended,
                      "mean_lp": lp[i, :end].mean().item() if end else -1e9})
    return cands

def gen_sentence_level(item, chooser):
    rng = item_rng(item, chooser)
    ids = encode_prefix(item["prefix"]); g0 = len(ids)
    calls = 0
    while len(ids) - g0 < MAX_NEW_TOKENS:
        cands = sentence_candidates(ids, MAX_NEW_TOKENS - (len(ids) - g0))
        if chooser == "jev" and len(cands) >= 2:
            texts = [display_key(c["text"]) for c in cands]
            if SENT_OPTION_STYLE == "letters":
                keys = [chr(65 + i) for i in range(len(cands))]
                k = jev_pick(item["prefix"] + tok.decode(ids[g0:]), keys, SENT_INSTR, rng, descriptions=texts)
            else:
                k = jev_pick(item["prefix"] + tok.decode(ids[g0:]), texts, SENT_INSTR, rng)
            calls += 1
        elif chooser == "selfrank":
            k = max(range(len(cands)), key=lambda i: cands[i]["mean_lp"])
        else:
            k = rng.randrange(len(cands))
        c = cands[k]
        ids = ids + c["tokens"]
        if c["ended"] or not c["tokens"]:
            break
    gen = ids[g0:g0 + MAX_NEW_TOKENS]
    return {"generation": tok.decode(gen), "gen_tokens": len(gen), "jev_calls": calls}

BASELINES = {
    "greedy":      dict(do_sample=False),
    "nucleus":     dict(do_sample=True, top_p=0.95, top_k=0),
    "contrastive": dict(penalty_alpha=0.6, top_k=5),
}
LOOP_METHODS = {
    "jev_word":               lambda it: gen_word_level(it, "jev"),
    "ctrl_word_random":       lambda it: gen_word_level(it, "random"),
    "jev_sentence":           lambda it: gen_sentence_level(it, "jev"),
    "ctrl_sentence_random":   lambda it: gen_sentence_level(it, "random"),
    "ctrl_sentence_selfrank": lambda it: gen_sentence_level(it, "selfrank"),
}

@torch.no_grad()
def baseline_batch(batch, kw):
    seqs = [encode_prefix(it["prefix"]) for it in batch]
    L = max(map(len, seqs))
    x = torch.full((len(seqs), L), EOS, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, L - len(s):] = torch.tensor(s); m[i, L - len(s):] = 1
    with GPU:
        out = gen_model.generate(input_ids=x.to(DEVICE), attention_mask=m.to(DEVICE), max_new_tokens=MAX_NEW_TOKENS,
                                 pad_token_id=EOS, eos_token_id=EOS, **kw)
    out = out.cpu()
    with GPU: mps_trim()
    res = []
    for row in out[:, L:].tolist():
        if EOS in row: row = row[:row.index(EOS)]
        res.append({"generation": tok.decode(row), "gen_tokens": len(row), "jev_calls": 0})
    return res

# %% cell 9
# --- Runner (saves every output as it is produced, so an interrupted run resumes) --
def method_path(name):
    tag = _gate_cfg if "word" in name else _sent_cfg if "sentence" in name else "base"
    return os.path.join(RUN_DIR, f"{name}__{hashlib.md5(tag.encode()).hexdigest()[:6]}.jsonl")

def run_method(name, items):
    path = method_path(name)
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for l in f:
                if l.strip():
                    r = json.loads(l); done[r["id"]] = r
    todo = [it for it in items if it["id"] not in done]
    if not todo:
        print(f"{name}: all {len(items)} outputs already saved"); return [done[it["id"]] for it in items]
    wlock = threading.Lock(); t0 = time.time(); j0 = JEV.snapshot(); n_done = 0

    def save(it, r):
        nonlocal n_done
        r.update(id=it["id"], seconds=round(r.get("seconds", 0), 2))
        with wlock:
            with open(path, "a") as f: f.write(json.dumps(r) + "\n")
            done[it["id"]] = r; n_done += 1
            if n_done % max(1, len(todo) // 10) == 0 or n_done == len(todo):
                el = time.time() - t0
                print(f"  {name}: {n_done}/{len(todo)}  {el/60:.1f} min elapsed, ~{el/n_done*(len(todo)-n_done)/60:.1f} min left"
                      f"  | Jev calls so far {JEV.snapshot()['calls'] - j0['calls']}")

    print(f"{name}: generating {len(todo)} outputs ...")
    if name in BASELINES:
        torch.manual_seed(SEED)
        # contrastive search keeps a KV cache for batch x top_k sequences (about 14 GB at batch 16 on gpt2-xl), so it gets a small batch
        bs = int(os.environ.get("NB_CONTRASTIVE_BATCH", 2)) if name == "contrastive" else BATCH_SIZE
        for b in range(0, len(todo), bs):
            chunk = todo[b:b + bs]; tb = time.time()
            for it, r in zip(chunk, baseline_batch(chunk, BASELINES[name])):
                r["seconds"] = (time.time() - tb) / len(chunk); save(it, r)
    else:
        fn = LOOP_METHODS[name]
        def work(it):
            ts = time.time(); r = fn(it); r["seconds"] = time.time() - ts; return it, r
        with ThreadPoolExecutor(max_workers=JEV_CONCURRENCY) as ex:
            for fut in as_completed([ex.submit(work, it) for it in todo]):
                save(*fut.result())
    j1 = JEV.snapshot()
    if j1["calls"] > j0["calls"]:
        n = j1["calls"] - j0["calls"]
        print(f"  {name}: {n} Jev calls, mean latency {(j1['lat_sum']-j0['lat_sum'])/n*1000:.0f} ms, "
              f"{j1['input_tokens']-j0['input_tokens']:,} input tokens (~${(j1['input_tokens']-j0['input_tokens'])*JEV_PRICE_PER_M_INPUT/1e6:.4f})")
    return [done[it["id"]] for it in items]

OUTPUTS = {}
for m in METHODS:
    OUTPUTS[m] = run_method(m, ITEMS)

# %% cell 11
def step_summary(name):
    rows = [s for r in OUTPUTS.get(name, []) for s in r.get("steps", [])]
    if not rows: return None
    df = pd.DataFrame(rows, columns=["case", "p_top1", "margin", "topk_mass", "picked_rank"])
    out = df.groupby("case").agg(steps=("case", "size"), mean_p_top1=("p_top1", "mean"), mean_topk_mass=("topk_mass", "mean"))
    out["share_of_steps_%"] = (100 * out["steps"] / len(df)).round(1)
    asked = df.dropna(subset=["picked_rank"])
    if len(asked):
        out["picker_overrode_top1_%"] = asked.groupby("case")["picked_rank"].apply(lambda r: round(100 * (r > 0).mean(), 1))
    return out.reindex([c for c in ["confident", "contested_diffuse", "contested_concentrated", "repeat"] if c in out.index]).round(3)

for name in ["jev_word", "ctrl_word_random"]:
    s = step_summary(name)
    if s is not None:
        print(f"\n{name}"); print(s.to_string())

# %% cell 13
import mauve, faiss
faiss.omp_set_num_threads(1)   # multi-threaded faiss k-means segfaults next to torch on macOS

def simctg_diversity(texts):
    """Port of simctg.evaluation.measure_repetition_and_diversity (returns rep-2, rep-3, rep-4, diversity, all in %)."""
    uniq = {2: 0, 3: 0, 4: 0}; tot = {2: 0, 3: 0, 4: 0}
    for t in texts:
        w = t.strip("\n").strip().split()
        for n in (2, 3, 4):
            grams = [" ".join(w[i:i + n]) for i in range(len(w) - n)]   # same off-by-one as the original
            uniq[n] += len(set(grams)); tot[n] += len(grams)
    rep = {n: round(100 * (1 - uniq[n] / tot[n]), 2) if tot[n] else 0.0 for n in (2, 3, 4)}
    div = (1 - rep[2] / 100) * (1 - rep[3] / 100) * (1 - rep[4] / 100)
    return rep[2], rep[3], rep[4], round(100 * div, 2)

# GPT-2 featurizer used by MAUVE (final-layer state of the last token, as in mauve-text)
feat_model = (GPT2Model.from_pretrained("gpt2").to(DEVICE).eval() if MODE == "real" else tiny_gpt2(SEED + 1, lm_head=False))
_feat_cache = {}
@torch.no_grad()
def featurize(texts):
    out = []
    for t in texts:
        if t not in _feat_cache:
            ids = tok.encode(t)[:1024] or [EOS]
            h = feat_model(input_ids=torch.tensor([ids], device=DEVICE), output_hidden_states=True).hidden_states[-1]
            _feat_cache[t] = h[0, -1].float().cpu().numpy()
        out.append(_feat_cache[t])
    return np.stack(out)

def mauve_score(refs, preds):
    P, Q = [], []
    for ref, pred in zip(refs, preds):
        rt, pt = tok.encode(ref)[:128], tok.encode(pred)[:128]
        if min(len(rt), len(pt)) == 128:
            P.append(tok.decode(rt)); Q.append(tok.decode(pt))
    if len(P) < 10:
        return float("nan"), len(P)
    return 100 * mauve.compute_mauve(p_features=featurize(P), q_features=featurize(Q), verbose=False).mauve, len(P)

# Coherence model
if MODE == "real":
    if DEVICE == "mps":
        gen_model.to("cpu"); torch.mps.empty_cache()
    coh_tok = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
    coh_model = OPTForCausalLM.from_pretrained("facebook/opt-2.7b", torch_dtype=DTYPE).to(DEVICE).eval()
    COH_BOS = coh_tok.bos_token_id
else:
    torch.manual_seed(SEED + 2)
    coh_tok, COH_BOS = tok, EOS
    coh_model = OPTForCausalLM(OPTConfig(vocab_size=len(tok), hidden_size=64, ffn_dim=128, num_hidden_layers=2,
                                         num_attention_heads=2, word_embed_proj_dim=64, max_position_embeddings=2048,
                                         bos_token_id=EOS, eos_token_id=EOS, pad_token_id=EOS)).to(DEVICE).eval()

_coh_cache = {}
@torch.no_grad()
def coherence(prefix, pred):
    key = (prefix, pred)
    if key not in _coh_cache:
        ctx = [COH_BOS] + coh_tok.encode(prefix, add_special_tokens=False)
        p = coh_tok.encode(pred, add_special_tokens=False)
        if not p:
            _coh_cache[key] = None
        else:
            x = torch.tensor([ctx + p], device=DEVICE)
            lp = torch.log_softmax(coh_model(input_ids=x).logits[0, len(ctx) - 1:-1].float(), -1)
            _coh_cache[key] = lp.gather(-1, torch.tensor(p, device=DEVICE).unsqueeze(-1)).mean().item()
    return _coh_cache[key]

def score(name, items, gens, source, extra=None):
    refs = [it["reference"] for it in items]
    r2, r3, r4, div = simctg_diversity([g for g in gens])
    mv, n_mv = mauve_score(refs, gens)
    cohs = [c for c in (coherence(it["prefix"], g) for it, g in zip(items, gens) if g.strip()) if c is not None]
    row = {"method": name, "source": source, "n": len(items), "diversity": div, "rep-2": r2, "rep-4": r4,
           "MAUVE": round(mv, 2), "MAUVE_n": n_mv, "coherence": round(float(np.mean(cohs)), 3) if cohs else float("nan"),
           "gen_len": round(float(np.mean([len(tok.encode(g)) for g in gens])), 1)}
    row.update(extra or {})
    return row

# %% cell 14
# --- Published reference numbers + re-scoring the published outputs on your subset --
PUBLISHED = [  # Su & Xu (2022), Table 1, Wikitext column, GPT-2 XL, 1314 prompts
    ("greedy", 1.77, 4.91, -0.41), ("top-k (k=50)", 87.49, 81.00, -2.37), ("nucleus (p=0.95)", 92.16, 86.54, -3.03),
    ("typical (0.95)", 94.82, 86.07, -3.71), ("contrastive decoding (XL vs small)", 88.02, 91.46, -2.19),
    ("contrastive search (k=5, α=0.6)", 89.35, 77.97, -1.56)]
rows = [{"method": m, "source": "published (paper, n=1314)", "n": 1314, "diversity": d, "MAUVE": mv, "coherence": c}
        for m, d, mv, c in PUBLISHED]

def load_published_outputs():
    out = {}
    for label, rel in [("greedy", "data_from_CD_repo/wikitext/wikitext_greedy_gpt2-xl_256.jsonl"),
                       ("nucleus (p=0.95)", "data_from_CD_repo/wikitext/wikitext_p-0.95_gpt2-xl_256.jsonl"),
                       ("contrastive decoding (XL vs small)", "data_from_CD_repo/wikitext/wikitext_gpt2-0.8-t0.5_gpt2-xl_256.jsonl")]:
        rs = [json.loads(l)[0] for l in fetch(rel).splitlines() if l.strip()]
        out[label] = [r["gen_text"][len(r["prompt"]):] if r["gen_text"].startswith(r["prompt"]) else "" for r in rs]
    cs = json.loads(fetch("inference_results/gpt2-xl/wikitext/wikitext_contrastive_k_5_alpha_0.6_result.json"))
    out["contrastive search (k=5, α=0.6)"] = [c["generated_result"]["0"] for c in cs]
    return out

PUB_OUT = load_published_outputs()
calib_sets = [("published outputs, re-scored here", ITEMS)]
if CALIBRATE_FULL and MODE == "real":
    calib_sets.append(("published outputs, re-scored here (all 1314)", ALL_ITEMS))
for source, its in calib_sets:
    for label, gens in PUB_OUT.items():
        rows.append(score(label, its, [gens[it["id"]] for it in its], source))
        print("scored", source, "|", label)

# %% cell 15
# --- Score your runs ---------------------------------------------------------
for m in METHODS:
    outs = OUTPUTS[m]
    calls = [r.get("jev_calls", 0) for r in outs]
    steps = [s for r in outs for s in r.get("steps", [])]
    gated = [s for s in steps if s[0] in GATED]
    extra = {"jev_calls_per_prompt": round(float(np.mean(calls)), 1),
             "sec_per_prompt": round(float(np.mean([r.get("seconds", 0) for r in outs])), 2)}
    if steps:
        extra["gated_steps_%"] = round(100 * len(gated) / len(steps), 1)
    rows.append(score(m, ITEMS, [r["generation"] for r in outs], f"this run ({GEN_MODEL if MODE == 'real' else 'DRY RUN'})", extra))
    print("scored", m)

RESULTS = pd.DataFrame(rows)
if MODE == "real" and DEVICE == "mps":   # scoring is done: swap the generator back in for the fact track
    del coh_model; _coh_cache.clear(); torch.mps.empty_cache(); gen_model.to(DEVICE)
RESULTS.to_csv(os.path.join(RUN_DIR, "results.csv"), index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
cols = ["method", "source", "n", "diversity", "MAUVE", "MAUVE_n", "coherence", "rep-2", "rep-4", "gen_len",
        "jev_calls_per_prompt", "gated_steps_%", "sec_per_prompt"]
print(RESULTS[[c for c in cols if c in RESULTS.columns]].to_string(index=False))
if MODE == "dry_run":
    print("\nDRY RUN: these numbers come from a random model and a fake Jev. They only show that every step runs.")

# %% cell 17
# --- Side-by-side samples ------------------------------------------------------
show = [m for m in ["greedy", "contrastive", "jev_word", "ctrl_word_random", "jev_sentence", "ctrl_sentence_random"] if m in OUTPUTS]
for k in range(min(3, len(ITEMS))):
    print("=" * 100); print("PROMPT:", ITEMS[k]["prefix"]); print("-" * 100)
    for m in show:
        print(f"[{m}]", OUTPUTS[m][k]["generation"][:600].replace("\n", " ⏎ ")); print()

# %% cell 19
FALLBACK_FACTS = [
    ("The capital city of Australia is", "Canberra"), ("The capital city of Canada is", "Ottawa"),
    ("The capital city of Turkey is", "Ankara"), ("The capital city of Brazil is", "Brasilia"),
    ("The capital city of Nigeria is", "Abuja"), ("The capital city of Pakistan is", "Islamabad"),
    ("The capital city of Switzerland is", "Bern"), ("The capital city of New Zealand is", "Wellington"),
    ("The capital city of Morocco is", "Rabat"), ("The capital city of Vietnam is", "Hanoi"),
    ("The official language of Brazil is", "Portuguese"), ("The currency of Japan is the", "yen"),
    ("The currency of the United Kingdom is the", "pound"), ("The currency of India is the", "rupee"),
    ("The Mona Lisa was painted by", "Leonardo"), ("Romeo and Juliet was written by William", "Shakespeare"),
    ("The theory of general relativity was developed by Albert", "Einstein"), ("The chemical symbol for gold is", "Au"),
    ("The chemical symbol for sodium is", "Na"), ("The largest planet in the Solar System is", "Jupiter"),
    ("The planet closest to the Sun is", "Mercury"), ("Water is made of hydrogen and", "oxygen"),
    ("The Great Pyramid of Giza is located in", "Egypt"), ("The Taj Mahal is located in the country of", "India"),
    ("The Amazon River flows mainly through", "Brazil"), ("The first person to walk on the Moon was Neil", "Armstrong"),
    ("Pride and Prejudice was written by Jane", "Austen"), ("The longest river in Africa is the", "Nile"),
    ("Tokyo is the capital of", "Japan"), ("Berlin is the capital of", "Germany"), ("Lisbon is the capital of", "Portugal"),
    ("The Kremlin is located in", "Moscow"), ("The Colosseum is located in", "Rome"),
    ("Toyota is a car company based in", "Japan"), ("Volkswagen is a car company based in", "Germany"),
    ("Barack Obama was born in the U.S. state of", "Hawaii"), ("The Eiffel Tower is located in", "Paris"),
    ("The Ninth Symphony was composed by Ludwig van", "Beethoven"), ("Microsoft was founded by Bill", "Gates"),
    ("The capital city of Kenya is", "Nairobi"),
]

def load_facts(n):
    try:
        cp = os.path.join(CACHE_DIR, "counterfact.json")
        if not os.path.exists(cp):
            r = requests.get(COUNTERFACT_URL, timeout=120); r.raise_for_status()
            with open(cp, "w") as f: f.write(r.text)
        with open(cp) as f: data = json.load(f)
        facts = [(d["requested_rewrite"]["prompt"].format(d["requested_rewrite"]["subject"]),
                  d["requested_rewrite"]["target_true"]["str"]) for d in data]
        print(f"CounterFact: {len(facts)} facts loaded")
    except Exception as e:
        print(f"Could not load CounterFact ({type(e).__name__}); using the {len(FALLBACK_FACTS)} built-in facts instead")
        facts = FALLBACK_FACTS
    return random.Random(SEED).sample(facts, min(n, len(facts)))

def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)

FACT_INSTR = ("Choose the word that should come immediately after the text so that the statement is true. "
              "If several options would be true, choose the most specific correct one.")

def fact_one(i, prompt, gold):
    ids = tok.encode(prompt)
    with GPU, torch.no_grad():
        probs = torch.softmax(gen_model(input_ids=torch.tensor([ids], device=DEVICE)).logits[0, -1].float(), -1).cpu()
    case, _ = gate_case(ids, probs, len(ids))
    cands = word_candidates(ids, probs)
    keys = [c["key"] for c in cands]
    g = norm(gold.split()[0])
    hit = [norm(k) == g for k in keys]
    rng = random.Random(SEED * 1000 + i)
    jk = jev_pick(prompt, keys, FACT_INSTR, rng) if len(keys) >= 2 else 0
    return {"prompt": prompt, "gold": gold, "case": case, "gpt2": keys[0], "jev": keys[jk],
            "gpt2_ok": hit[0], "jev_ok": hit[jk], "in_top_k": any(hit),
            "gated_ok": hit[jk] if case in GATED else hit[0]}

if RUN_FACTS:
    facts = load_facts(N_FACTS)
    fpath = os.path.join(RUN_DIR, f"facts__{hashlib.md5(_gate_cfg.encode()).hexdigest()[:6]}_{len(facts)}.jsonl")
    if os.path.exists(fpath):
        FACTS = [json.loads(l) for l in open(fpath)]
    else:
        with ThreadPoolExecutor(max_workers=JEV_CONCURRENCY) as ex:
            FACTS = list(ex.map(lambda a: fact_one(*a), [(i, p, g) for i, (p, g) in enumerate(facts)]))
        with open(fpath, "w") as f:
            for r in FACTS: f.write(json.dumps(r) + "\n")
    F = pd.DataFrame(FACTS)
    n = len(F)
    print(f"\nFacts: {n} prompts (answer counted correct if the chosen word equals the first word of the true answer)")
    print(f"  GPT-2 top word correct:                      {100*F.gpt2_ok.mean():5.1f}%")
    print(f"  Jev picking from GPT-2's top {TOP_K} words:        {100*F.jev_ok.mean():5.1f}%")
    print(f"  Jev only when the gate fires, else GPT-2:    {100*F.gated_ok.mean():5.1f}%")
    print(f"  Ceiling (true answer is among the top {TOP_K}):   {100*F.in_top_k.mean():5.1f}%")
    print(f"  When the answer was in the list, Jev chose it {100*F[F.in_top_k].jev_ok.mean():.1f}% of the time vs GPT-2 {100*F[F.in_top_k].gpt2_ok.mean():.1f}%")
    by = F.groupby("case").agg(prompts=("case", "size"), gpt2_acc=("gpt2_ok", "mean"), jev_acc=("jev_ok", "mean"),
                               answer_in_top_k=("in_top_k", "mean"))
    print("\nBy gate case:"); print((by.assign(**{c: (100 * by[c]).round(1) for c in ["gpt2_acc", "jev_acc", "answer_in_top_k"]})).to_string())
    print("\nExamples where Jev and GPT-2 disagree:")
    print(F[F.gpt2 != F.jev][["prompt", "gold", "gpt2", "jev"]].head(12).to_string(index=False))
    F.to_csv(os.path.join(RUN_DIR, "facts.csv"), index=False)

# %% cell 20
j = JEV.snapshot()
print(f"Total Jev calls this session: {j['calls']:,} | input tokens {j['input_tokens']:,} | "
      f"est. cost ${j['input_tokens']*JEV_PRICE_PER_M_INPUT/1e6:.4f} | errors {j['errors']}")
print("Saved:", RUN_DIR)
