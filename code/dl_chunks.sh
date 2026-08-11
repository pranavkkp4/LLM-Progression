#!/bin/bash
# Download Qwen2.5-0.5B-Instruct weights in ranged chunks (server caps each
# response at 100 MiB) and assemble into /tmp. Kept in the repo because /tmp
# can be wiped between sessions.
URL="https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/resolve/master/model.safetensors"
TMP=/tmp/model_full.safetensors
TOTAL=988097824
CHUNK=99000000
rm -f "$TMP"; touch "$TMP"
POS=0
while [ $POS -lt $TOTAL ]; do
  END=$((POS + CHUNK - 1)); [ $END -ge $TOTAL ] && END=$((TOTAL - 1))
  WANT=$((END - POS + 1))
  ok=0
  for attempt in 1 2 3 4 5; do
    curl -sL --max-time 600 -r ${POS}-${END} -o /tmp/chunk_part "$URL"
    GOT=$(stat -c%s /tmp/chunk_part 2>/dev/null || echo 0)
    if [ "$GOT" = "$WANT" ]; then ok=1; break; fi
    sleep 2
  done
  [ $ok -ne 1 ] && { echo "FAILED at $POS got $GOT want $WANT"; exit 1; }
  cat /tmp/chunk_part >> "$TMP" || { echo "append failed"; exit 1; }
  POS=$((END + 1))
  echo "progress: $POS / $TOTAL"
done
FINAL=$(stat -c%s "$TMP")
echo "assembled $FINAL"
[ "$FINAL" = "$TOTAL" ] && echo "DONE-OK" || { echo "SIZE MISMATCH"; exit 1; }
