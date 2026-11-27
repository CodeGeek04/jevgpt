# Jev as a picker for GPT-2 XL decoding

Tests whether Jev (TypeSafe System One, `jev-latest`) improves GPT-2 XL text generation when it picks the next word or next sentence from GPT-2's own candidates.

Benchmark: WikiText-103 open-ended generation (32-token prompt, 256 new tokens), 100 prompts, seed 0. Fact track: 500 CounterFact prompts.

## Results (100 prompts)

| Method | Diversity | MAUVE | Coherence | Jev calls per prompt |
|---|---|---|---|---|
| greedy | 1.54 | 5.02 | -0.411 | 0 |
| nucleus (top-p 0.95) | 92.78 | 98.60 | -2.961 | 0 |
| contrastive search (k=5, alpha=0.6) | 93.60 | 94.59 | -1.576 | 0 |
| jev_sentence (Jev picks from 6 sampled sentences) | 92.10 | 91.52 | -2.666 | 12.3 |
| ctrl_sentence_random (random pick, same candidates) | 93.53 | 96.86 | -3.049 | 0 |
| ctrl_sentence_selfrank (GPT-2 picks its most likely) | 63.16 | 73.93 | -1.776 | 0 |
| jev_word (Jev picks from top 8 words when the gate fires) | 15.41 | 32.53 | -1.191 | 135.7 |
| ctrl_word_random (GPT-2-weighted random pick, same gate) | 36.32 | 74.86 | -1.112 | 0 |

Paired coherence differences (10,000 bootstrap resamples):

| Comparison | Difference | 95% CI | Prompts where first is better |
|---|---|---|---|
| jev_sentence minus ctrl_sentence_random | +0.383 | +0.299 to +0.468 | 78 of 100 |
| jev_sentence minus nucleus | +0.295 | +0.189 to +0.395 | 77 of 100 |
| jev_sentence minus contrastive | -1.090 | -1.179 to -0.996 | 1 of 100 |
| jev_word minus ctrl_word_random | -0.078 | -0.153 to -0.006 | 39 of 100 |

Fact track (500 CounterFact prompts):

| Picker | Correct |
|---|---|
| GPT-2 top word | 9.8% |
| Jev picks from GPT-2's top 8 | 23.0% |
| Jev only when the gate fires | 9.6% |
| Ceiling (answer is in the top 8) | 28.4% |

When the answer was in the list (142 prompts), Jev was right on 115 and GPT-2 on 49. Jev fixed 69 and broke 3. 52 of the 69 fixes replaced a filler word such as "the" with the entity.

## Findings

- Word-level picking does not work. It is worse than the random control on all three metrics.
- Sentence-level picking is more coherent than a random pick from the same candidates, but has lower MAUVE than that control and does not beat contrastive search.
- Jev finds the right fact in GPT-2's top 8 more than twice as often as GPT-2's own top word, but the gate used in the text track does not capture that gain.

The scorer was checked against the published outputs: greedy MAUVE 4.99 (paper 4.91), nucleus 86.42 (paper 86.54).

## Files

- `jev_gpt2_experiment.py`: the experiment (converted from `jev_gpt2_experiment.ipynb`, with Apple Silicon fixes and env-configurable settings).
- `run_full.sh`: runs the full 100-prompt experiment with a restart watchdog. Reads `TYPESAFE_API_KEY` from `../coldcall/.env`.
- `trace_examples.py`: records GPT-2 and Jev probabilities for every gated word decision (`logs/trace_examples.jsonl`).
- `paired_coherence.py`: per-prompt coherence and paired bootstrap.
- `jev_gpt2_runs/real_gpt2-xl_N100_T256_seed0/`: full run outputs (`results.csv`, `facts.csv`, generations per method).
- `jev_gpt2_runs/real_gpt2-xl_N20_T256_seed0/`: 20-prompt pilot.

## Setup

```
uv venv --python 3.12 .venv
uv pip install torch transformers==4.46.3 mauve-text faiss-cpu pandas requests
./run_full.sh
```

Benchmark data downloads into `jev_gpt2_runs/cache/` on first run.
