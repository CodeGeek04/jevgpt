# Can a picker fix GPT-2's mistakes?

GPT-2 XL often has the right next word among its top candidates and still ranks something else first. This experiment hands that choice to a second model, Jev, TypeSafe's fast classification model (System One API, `jev-latest`). Jev sees the text so far and a list of GPT-2's own candidates, and picks one. It never writes anything itself.

There are three tests. Each has a control that picks at random from the same candidates, so any gain comes from the picking and not from the candidates.

- **Facts** (500 CounterFact prompts): Jev picks from GPT-2's top 8 next words.
- **Words** (100 WikiText-103 prompts, 256 new tokens): Jev picks the next word whenever GPT-2 looks unsure.
- **Sentences** (the same 100 prompts): Jev picks the next sentence from 6 that GPT-2 sampled.

## What happened

- **Facts: the clear win.** GPT-2's top word is right 9.8% of the time. With Jev picking from the same top 8 words, it is right 23.0% of the time, out of a possible 28.4%. 52 of Jev's 69 fixes replaced a filler word such as "the" with the answer. GPT-2 had the fact. It just put a filler word first.
- **Words: does not work.** It is worse than the random control on diversity, MAUVE and coherence. Jev keeps choosing full stops: its text has 47.5 full stops per 100 words, against 6.0 for the control.
- **Sentences: partial.** Jev's text is more coherent than a random pick from the same candidates (better on 78 of 100 prompts), but it has lower MAUVE than that control and is well behind contrastive search.
- **The gate was wrong for facts.** The word method only calls Jev when GPT-2 is unsure and its top 8 words hold at least 70% of the probability. On facts, that rule keeps none of the gain (9.6%). Most fixes (56 of 69) came where GPT-2's probability was spread out, which is exactly where the gate does not ask.

## Results

### Text (100 WikiText-103 prompts)

All rows are scored the same way on the same 100 prompts, so compare them with each other.

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

### Facts (500 CounterFact prompts)

| Picker | Correct |
|---|---|
| GPT-2 top word | 9.8% |
| Jev picks from GPT-2's top 8 | 23.0% |
| Jev only when the gate fires | 9.6% |
| Ceiling (answer is in the top 8) | 28.4% |

When the answer was in the list (142 prompts), Jev was right on 115 and GPT-2 on 49. Jev fixed 69 and broke 3.

By what GPT-2's next-word probabilities looked like:

| GPT-2's next word | Prompts | Answer in top 8 | GPT-2 right | Jev right |
|---|---|---|---|---|
| confident | 110 | 32 | 15 | 27 |
| unsure, probability spread out | 362 | 105 | 31 | 86 |
| unsure, top 8 hold 70% or more (the gate fires) | 28 | 5 | 3 | 2 |

## How it works

- **Benchmark.** Prompts, metrics and baseline settings follow Su and Xu (2022), "An Empirical Study On Contrastive Search And Contrastive Decoding For Open-ended Text Generation", using the data in [their repo](https://github.com/yxuansu/Contrastive_Search_versus_Contrastive_Decoding). The scorer was checked against their published outputs: greedy MAUVE 4.99 (paper 4.91), nucleus 86.42 (paper 86.54). Facts come from [CounterFact](https://rome.baulab.info/) (Meng et al., 2022).
- **The gate (word method).** GPT-2 counts as unsure when its top word has under 40% probability, or its top two words are within 15 points. Jev is called when GPT-2 is unsure and its top 8 words hold at least 70% of the probability, or when GPT-2's top word would repeat a 4-token sequence it already wrote. At most 150 calls per prompt. "[end of text]" is never offered as an option, so lengths stay comparable.
- **Sentences.** GPT-2 samples 6 candidate sentences of up to 40 tokens. Jev sees them as options A to F and picks one.

## Files

- `jev_gpt2_experiment.py`: the experiment (converted from `jev_gpt2_experiment.ipynb`, with Apple Silicon fixes and settings you can change through environment variables).
- `run_full.sh`: runs the full 100-prompt experiment and restarts it if it hangs. Outputs are saved per prompt, so a restart resumes.
- `trace_examples.py`: records GPT-2 and Jev probabilities for every gated word decision (`logs/trace_examples.jsonl`).
- `paired_coherence.py`: per-prompt coherence and the paired bootstrap.
- `jev_gpt2_runs/real_gpt2-xl_N100_T256_seed0/`: the full run (`results.csv`, `facts.csv`, and every generation per method).
- `jev_gpt2_runs/real_gpt2-xl_N20_T256_seed0/`: the 20-prompt pilot.
- `logs/`: logs from the dry run, the pilots and the full run.

## Running it

You need a TypeSafe API key. Export `TYPESAFE_API_KEY`, or put `TYPESAFE_API_KEY=...` in a `.env` file in this folder (it is gitignored).

```
uv venv --python 3.12 .venv
uv pip install torch transformers==4.46.3 mauve-text faiss-cpu pandas requests
./run_full.sh
```

`run_full.sh` is written for macOS (it uses `caffeinate` and BSD `stat`). The experiment itself runs anywhere PyTorch does. Benchmark data downloads into `jev_gpt2_runs/cache/` on first run.

## License

MIT. See [LICENSE](LICENSE).
