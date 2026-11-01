#!/bin/bash
# The three authors' books from Project Gutenberg, keyless, via curl. gutendex.com times
# out from some networks, so this uses Gutenberg's own catalogue feed instead.
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
D="$P/data/gutenberg"
PY=${PY:-python}
mkdir -p "$D"

if [ ! -s "$D/pg_catalog.csv" ]; then
  echo "=== catalogue ==="
  curl -sL --retry 4 --retry-delay 2 --max-time 180 \
    -o "$D/pg_catalog.csv.gz" "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
  gunzip -f "$D/pg_catalog.csv.gz"
  wc -l "$D/pg_catalog.csv"
fi

echo
echo "=== selected books ==="
"$PY" -I "$P/filter_catalog.py" > "$D/books.tsv"
cat "$D/books.tsv"

echo
echo "=== downloading ==="
ok=0; fail=0
while IFS=$'\t' read -r code num title; do
  mkdir -p "$D/$code"
  f="$D/$code/$num.txt"
  if [ -s "$f" ] && [ "$(stat -c%s "$f")" -gt 1000 ]; then ok=$((ok+1)); continue; fi
  for u in "https://www.gutenberg.org/cache/epub/$num/pg$num.txt" \
           "https://www.gutenberg.org/files/$num/$num-0.txt" \
           "https://www.gutenberg.org/files/$num/$num.txt"; do
    code_http=$(curl -sL --retry 3 --retry-delay 2 --retry-connrefused --max-time 180 \
      -o "$f" -w "%{http_code}" "$u")
    if [ "$code_http" = "200" ] && [ "$(stat -c%s "$f")" -gt 1000 ]; then break; fi
    rm -f "$f"
  done
  if [ -s "$f" ]; then
    ok=$((ok+1))
    printf "  %-4s %-7s %9d  %s\n" "$code" "$num" "$(stat -c%s "$f")" "$title"
  else
    fail=$((fail+1))
    printf "  %-4s %-7s   MISSING  %s\n" "$code" "$num" "$title"
  fi
  sleep 0.3
done < "$D/books.tsv"

echo
echo "=== totals ==="
echo "downloaded $ok, missing $fail"
for c in EAP HPL MWS; do
  n=$(ls "$D/$c" 2>/dev/null | wc -l)
  b=$(du -sb "$D/$c" 2>/dev/null | cut -f1)
  echo "$c: $n files, $((b/1000000)) MB"
done
