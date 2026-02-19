#!/usr/bin/env bash
# Live wash-trade watcher — runs detection every N minutes on today's OKX BTC/USDT feed
# Usage: ./watch_live.sh [interval_minutes] [lookback_minutes]

INTERVAL=${1:-5}
LOOKBACK=${2:-30}
DATA_DIR="${DATA_DIR:-${3:-}}"
PROJECT="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$DATA_DIR" ]; then
    echo "Error: DATA_DIR not set. Usage: ./watch_live.sh [interval] [lookback] [data_dir]"
    echo "  or: DATA_DIR=/path/to/data ./watch_live.sh"
    exit 1
fi
NORM_DB="/tmp/wash_live_norm.db"
CONFIG="$PROJECT/learning/tuned_config.json"
export PYTHONPATH="$PROJECT/src"

echo "=== WASH DETECTOR LIVE WATCH ==="
echo "  Symbol:   OKX BTC/USDT"
echo "  Interval: every ${INTERVAL}m"
echo "  Lookback: last ${LOOKBACK}m"
echo "  Config:   tuned_config.json"
echo "  Press Ctrl+C to stop"
echo ""

while true; do
    # Find today's DB
    TODAY=$(date -u +%Y%m%d)
    SOURCE_DB="$DATA_DIR/btcusdt_${TODAY}.db"

    if [ ! -f "$SOURCE_DB" ]; then
        echo "[$(date -u '+%H:%M:%S')] No DB for today ($TODAY), waiting..."
        sleep $((INTERVAL * 60))
        continue
    fi

    # Ingest → normalized (silently)
    python3 -m wash_detector.cli ingest \
        --source-db "$SOURCE_DB" \
        --output-db "$NORM_DB" \
        --overwrite > /dev/null 2>&1

    if [ $? -ne 0 ]; then
        echo "[$(date -u '+%H:%M:%S')] Ingest failed, retrying in ${INTERVAL}m..."
        sleep $((INTERVAL * 60))
        continue
    fi

    # Detect
    echo "──────────────────────────────────────────── $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    python3 -m wash_detector.cli detect \
        --normalized-db "$NORM_DB" \
        --lookback-minutes "$LOOKBACK" \
        --config "$CONFIG" 2>&1

    sleep $((INTERVAL * 60))
done
