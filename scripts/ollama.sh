#!/bin/bash
# Local Ollama server for `detector.py label --mode ollama`; binary and models live under $OLLAMA_DIR.
#
#   bash scripts/ollama.sh install               # once, login node (needs internet)
#   bash scripts/ollama.sh pull qwen3.8:27b      # once, login node (needs internet)
#   bash scripts/ollama.sh serve                 # GPU node, before labeling
#   bash scripts/ollama.sh stop                  # after labeling, frees the GPU
#
# Port clash with another server on the node: export OLLAMA_HOST=127.0.0.1:<port> for both this script and detector.py.

set -euo pipefail

OLLAMA_DIR=${OLLAMA_DIR:-$HOME/scratch/ollama}
export OLLAMA_MODELS=$OLLAMA_DIR/models
export OLLAMA_HOST=${OLLAMA_HOST:-127.0.0.1:11434}
BIN=$OLLAMA_DIR/bin/ollama
PID_FILE=$OLLAMA_DIR/serve.pid

is_up() { curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null; }

serve() {
    if is_up; then echo "[ollama] already serving on $OLLAMA_HOST"; return; fi
    nohup "$BIN" serve >"$OLLAMA_DIR/serve.log" 2>&1 &
    echo $! >"$PID_FILE"
    for _ in $(seq 60); do is_up && { echo "[ollama] serving on $OLLAMA_HOST (log $OLLAMA_DIR/serve.log)"; return; }; sleep 1; done
    echo "[ollama] server did not come up; see $OLLAMA_DIR/serve.log"; exit 1
}

stop() {
    [[ -f "$PID_FILE" ]] && kill "$(cat "$PID_FILE")" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "[ollama] stopped"
}

case "${1:-}" in
    install)
        mkdir -p "$OLLAMA_DIR" "$OLLAMA_MODELS"
        curl -fL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst \
            | tar --zstd -x -C "$OLLAMA_DIR"
        "$BIN" --version
        ;;
    pull)
        started=0
        is_up || { serve; started=1; }
        "$BIN" pull "${2:?usage: $0 pull <model>}"
        [[ $started == 1 ]] && stop
        ;;
    serve) serve ;;
    stop) stop ;;
    *) echo "usage: $0 install | pull <model> | serve | stop"; exit 1 ;;
esac
