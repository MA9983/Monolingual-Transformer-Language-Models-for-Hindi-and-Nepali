#!/bin/bash
# Keeps hindi news scraping running with no idle gap: relaunches
# collect_manual_news.py immediately every time it exits (whether it
# stopped because it exhausted a discovery batch, hit its token target,
# or crashed), instead of relying on periodic external checks that can
# leave the scraper idle for a long stretch between a stop and the next
# restart.
cd "$(dirname "$0")/../.."
VENV_PY=".venv/bin/python3"
LOG="logs/hindi_scrape.log"

while true; do
    "$VENV_PY" hindi/data/collect_manual_news.py >> "$LOG" 2>&1
    # If the target has been reached, stop looping instead of spinning
    # forever re-launching a collector that will immediately no-op.
    reached=$("$VENV_PY" -c "
import json
try:
    with open('hindi/data/raw/manual/news_scrape/progress.json') as f:
        p = json.load(f)
    print('yes' if p.get('total_chars', 0) >= p.get('target_chars', 1) else 'no')
except Exception:
    print('no')
")
    if [ "$reached" = "yes" ]; then
        echo "[watchdog] target reached, stopping watchdog loop" >> "$LOG"
        break
    fi
    sleep 2
done
