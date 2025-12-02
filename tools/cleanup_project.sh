#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/vipuser/MindEyeV2_Project"
REPORT="/tmp/cleanup_report.txt"
rm -f "$REPORT"
echo -e "CLEANUP REPORT\nproject=$PROJECT" > "$REPORT"
echo "--- BEFORE ---" >> "$REPORT"
df -h "$PROJECT" | tee -a "$REPORT"
FS_LINE=$(df -B1 "$PROJECT" | tail -n1)
FS_USED=$(awk '{print $3}' <<<"$FS_LINE")
FS_AVAIL=$(awk '{print $4}' <<<"$FS_LINE")
echo "FS_USED_BYTES=$FS_USED" >> "$REPORT"
echo "FS_AVAIL_BYTES=$FS_AVAIL" >> "$REPORT"
du -sh "$PROJECT" | tee -a "$REPORT"
echo 'Top 80 large files/dirs under project:' >> "$REPORT"
(du -ah "$PROJECT" 2>/dev/null || true) | sort -rh | head -n 80 >> "$REPORT" || true

CACHE_DIR="$PROJECT/cache"
RUNS_LINK="$PROJECT/runs"
RUNS_REAL=$(readlink -f "$RUNS_LINK" 2>/dev/null || true)
UNCLIP="$CACHE_DIR/unclip6_epoch0_step110000.ckpt"
echo '--- IDENTIFY TARGETS ---' >> "$REPORT"
echo "CACHE_DIR=$CACHE_DIR" >> "$REPORT"
echo "RUNS_LINK=$RUNS_LINK" >> "$REPORT"
echo "RUNS_REAL=$RUNS_REAL" >> "$REPORT"
echo "UNCLIP_PATH=$UNCLIP" >> "$REPORT"
if [ -e "$UNCLIP" ]; then
  du -sh "$UNCLIP" >> "$REPORT"
else
  echo 'unclip not found' >> "$REPORT"
fi
if [ -d "$CACHE_DIR" ]; then
  du -sh "$CACHE_DIR" >> "$REPORT"
else
  echo 'cache dir not found' >> "$REPORT"
fi
if [ -n "$RUNS_REAL" ] && [ -d "$RUNS_REAL" ]; then
  du -sh "$RUNS_REAL" >> "$REPORT"
else
  echo 'runs real path not found or not dir' >> "$REPORT"
fi

printf '--- DELETION ACTIONS ---\n' >> "$REPORT"
TOTAL_FREED=0
# delete unclip if present
if [ -f "$UNCLIP" ]; then
  SIZE=$(stat -c%s "$UNCLIP" 2>/dev/null || echo 0)
  printf 'Deleting %s size=%s bytes\n' "$UNCLIP" "$SIZE" >> "$REPORT"
  rm -f "$UNCLIP" || true
  TOTAL_FREED=$((TOTAL_FREED + SIZE))
else
  printf 'No unclip file to delete\n' >> "$REPORT"
fi

# clear cache contents by removing all entries under cache (but leave the cache dir itself)
if [ -d "$CACHE_DIR" ]; then
  BEFORE_CACHE=$(du -sb "$CACHE_DIR" 2>/dev/null | cut -f1 || echo 0)
  printf 'Cache size before bytes=%s\n' "$BEFORE_CACHE" >> "$REPORT"
  # remove children safely
  find "$CACHE_DIR" -mindepth 1 -maxdepth 1 -print0 | while IFS= read -r -d '' item; do
    SZ=$(du -sb "$item" 2>/dev/null | cut -f1 || echo 0)
    printf 'Removing cache item: %s size=%s\n' "$item" "$SZ" >> "$REPORT"
    rm -rf "$item" || true
    TOTAL_FREED=$((TOTAL_FREED + SZ))
  done
else
  printf 'Cache dir not present; skipping cache cleanup\n' >> "$REPORT"
fi

# prune old runs (keep subj01_inference_run_final and modified within 7 days)
if [ -n "$RUNS_REAL" ] && [ -d "$RUNS_REAL" ]; then
  printf 'Pruning run subdirs in %s\n' "$RUNS_REAL" >> "$REPORT"
  for d in "$RUNS_REAL"/*; do
    [ -e "$d" ] || continue
    [ -d "$d" ] || continue
    name=$(basename "$d")
    if [ "$name" = "subj01_inference_run_final" ]; then
      printf 'Keeping protected %s\n' "$name" >> "$REPORT"
      continue
    fi
    # keep if modified within 7 days
    if find "$d" -maxdepth 0 -mtime -7 -print -quit | grep -q .; then
      printf 'Keeping recent %s\n' "$name" >> "$REPORT"
      continue
    fi
    SZ=$(du -sb "$d" 2>/dev/null | cut -f1 || echo 0)
    printf 'Deleting run dir %s size=%s\n' "$d" "$SZ" >> "$REPORT"
    rm -rf "$d" || true
    TOTAL_FREED=$((TOTAL_FREED + SZ))
  done
else
  printf 'No runs real dir found or not a directory\n' >> "$REPORT"
fi

# prune train_logs conservatively: skip dirs containing 'subj01' and keep recently modified (<1 day)
TRAIN_LOGS_DIR="$PROJECT/train_logs"
if [ -d "$TRAIN_LOGS_DIR" ]; then
  printf 'Pruning train_logs in %s\n' "$TRAIN_LOGS_DIR" >> "$REPORT"
  for d in "$TRAIN_LOGS_DIR"/*; do
    [ -e "$d" ] || continue
    [ -d "$d" ] || continue
    name=$(basename "$d")
    # keep recent
    if find "$d" -maxdepth 0 -mtime -1 -print -quit | grep -q .; then
      printf 'Keeping recent train_logs %s\n' "$name" >> "$REPORT"
      continue
    fi
    if echo "$name" | grep -q 'subj01'; then
      printf 'Skipping %s (contains subj01)\n' "$name" >> "$REPORT"
      continue
    fi
    SZ=$(du -sb "$d" 2>/dev/null | cut -f1 || echo 0)
    printf 'Deleting train_logs dir %s size=%s\n' "$d" "$SZ" >> "$REPORT"
    rm -rf "$d" || true
    TOTAL_FREED=$((TOTAL_FREED + SZ))
  done
else
  printf 'No train_logs dir found\n' >> "$REPORT"
fi

printf 'TOTAL_FREED_BYTES=%s\n' "$TOTAL_FREED" >> "$REPORT"
printf '--- AFTER ---\n' >> "$REPORT"
df -h "$PROJECT" | tee -a "$REPORT"
FS_LINE2=$(df -B1 "$PROJECT" | tail -n1)
FS_USED2=$(awk '{print $3}' <<<"$FS_LINE2")
FS_AVAIL2=$(awk '{print $4}' <<<"$FS_LINE2")
printf 'FS_USED_BYTES_AFTER=%s\nFS_AVAIL_BYTES_AFTER=%s\n' "$FS_USED2" "$FS_AVAIL2" >> "$REPORT"
if [ -n "$FS_AVAIL" ] && [ -n "$FS_AVAIL2" ]; then
  DIFF=$((FS_AVAIL2 - FS_AVAIL))
  printf 'AVAIL_DELTA_BYTES=%s\n' "$DIFF" >> "$REPORT"
fi

du -sh "$PROJECT" | tee -a "$REPORT"
printf 'Top 40 after:\n' >> "$REPORT"
(du -ah "$PROJECT" 2>/dev/null || true) | sort -rh | head -n 40 >> "$REPORT" || true

# show report
cat "$REPORT"
