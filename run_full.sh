#!/bin/zsh
# Runs the 100-prompt experiment and restarts it if it hangs. Outputs are saved per prompt, so a restart resumes.
# The Apple GPU backend occasionally blocks forever on a command buffer; a stale log is how that shows up.
cd "$(dirname "$0")"
export TYPESAFE_API_KEY="$(grep -E '^TYPESAFE_API_KEY=' ../coldcall/.env | head -1 | cut -d= -f2- | tr -d "\"' \r")"
[ -n "$TYPESAFE_API_KEY" ] || { echo "no API key"; exit 1; }
export NB_MODE=real NB_N_PROMPTS=100 NB_N_FACTS=500 NB_JEV_CONCURRENCY=8 NB_ALLOW_EOS_OPTION=0 NB_MAX_JEV_CALLS=100000
export NB_METHODS=greedy,nucleus,contrastive,jev_sentence,ctrl_sentence_random,ctrl_sentence_selfrank,jev_word,ctrl_word_random
LOG=logs/full_n100.log; STALE=${STALE_SECONDS:-900}
for attempt in 1 2 3 4 5 6 7 8; do
  echo "=== attempt $attempt $(date) ===" >> $LOG
  caffeinate -i .venv/bin/python -u jev_gpt2_experiment.py >> $LOG 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    age=$(( $(date +%s) - $(stat -f %m $LOG) ))
    if [ $age -gt $STALE ]; then
      echo "=== watchdog: no output for ${age}s, restarting $(date) ===" >> $LOG
      kill $pid; sleep 5; kill -9 $pid 2>/dev/null; break
    fi
  done
  wait $pid; rc=$?
  [ $rc -eq 0 ] && { echo "=== finished $(date) ===" >> $LOG; exit 0; }
done
echo "=== gave up after 8 attempts ===" >> $LOG; exit 1
