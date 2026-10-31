#!/bin/bash
# The LLM leg through the Claude Code CLI, so no API key is needed. 500 classification
# calls plus ~284 probes. Replies are cached, so an interrupted run resumes for free.
#
#   bash scripts/run_llm.sh                                # opus, clean prompt, 500
#   bash scripts/run_llm.sh claude-opus-5-5 dirty 120      # A/B with CLAUDE.md injected
set -u
MODEL=${1:-claude-opus-5-5}
MODE=${2:-clean}
N=${3:-500}
P="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PY:-python}
BIN=${CLAUDE_BIN:-claude}

# A scratch cwd and a config dir of its own, so no CLAUDE.md or project context
# reaches the prompt. Mode "dirty" leaves the real environment in place for the A/B.
FH="$P/.cli-home"
TS="${TMPDIR:-/tmp}/spooky-scratch"
mkdir -p "$TS" "$FH/.claude"
ln -sf "$HOME/.claude/.credentials.json" "$FH/.claude/.credentials.json"
[ -f "$FH/.claude/.claude.json" ] || echo '{}' > "$FH/.claude/.claude.json"

if [ "$MODE" = "clean" ]; then
  ENVARGS=(--cli-env "HOME=$FH" --cli-env "CLAUDE_CONFIG_DIR=$FH/.claude")
  EXTRA=""
else
  ENVARGS=()
  EXTRA="--no-probe"
fi

cd "$P" || exit 1
"$PY" -I spooky.py \
  --data data/gh_train.csv \
  --backend cli --claude-bin "$BIN" --cli-cwd "$TS" "${ENVARGS[@]}" \
  --model "$MODEL" \
  --n-llm "$N" --sample-pool 500 --workers 12 \
  $EXTRA \
  --cache "cache_${MODEL}_${MODE}.jsonl" \
  --out "predictions_${MODEL}_${MODE}_${N}.csv"
