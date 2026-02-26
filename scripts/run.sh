#!/usr/bin/env bash
#
# run.sh — Unified launcher with 3 connectivity modes.
#
# Usage:
#   ./scripts/run.sh                    # Interactive mode selection
#   ./scripts/run.sh --local            # Local (localhost only)
#   ./scripts/run.sh --lan              # LAN/WiFi (self-signed HTTPS)
#   ./scripts/run.sh --tunnel           # Cellular (Cloudflare Tunnel)
#   ./scripts/run.sh --check            # Validate environment only
#   ./scripts/run.sh --port 9000        # Override port (default: 8090)
#   ./scripts/run.sh --example minimal  # Run minimal-voice-app instead of with-admin
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Source library modules
source "$REPO_ROOT/scripts/lib/colors.sh"
source "$REPO_ROOT/scripts/lib/network.sh"
source "$REPO_ROOT/scripts/lib/cert.sh"
source "$REPO_ROOT/scripts/lib/health.sh"
source "$REPO_ROOT/scripts/lib/tunnel.sh"

# ── Defaults ─────────────────────────────────────────────────
PORT="${PORT:-8090}"
EXAMPLE="with-admin"
MODE=""
CHECK_ONLY=false

# PIDs to clean up
SERVER_PID=""
CAFFEINATE_PID=""
TAIL_PID=""

# ── Parse flags ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)            MODE="local"; shift ;;
        --lan)              MODE="lan"; shift ;;
        --tunnel)           MODE="tunnel"; shift ;;
        --check)            CHECK_ONLY=true; shift ;;
        --port)
            if [[ -z "${2:-}" || "$2" == --* ]]; then
                echo "Error: --port requires a numeric argument"; exit 1
            fi
            PORT="$2"; shift 2 ;;
        --example)
            if [[ -z "${2:-}" || "$2" == --* ]]; then
                echo "Error: --example requires an argument"; exit 1
            fi
            EXAMPLE="$2"; shift 2 ;;
        *)                  echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Resolve example name to directory
case "$EXAMPLE" in
    minimal|minimal-voice-app) EXAMPLE_DIR="examples/minimal-voice-app" ;;
    admin|with-admin)          EXAMPLE_DIR="examples/with-admin" ;;
    auth|with-auth)            EXAMPLE_DIR="examples/with-auth" ;;
    custom|custom-engine)      EXAMPLE_DIR="examples/custom-engine" ;;
    *)                         EXAMPLE_DIR="examples/$EXAMPLE" ;;
esac

if [ ! -f "$REPO_ROOT/$EXAMPLE_DIR/server.py" ]; then
    fail "Example not found: $EXAMPLE_DIR/server.py"
    exit 1
fi

VENV_PY="$REPO_ROOT/.venv/bin/python"
VENV_UVICORN="$REPO_ROOT/.venv/bin/uvicorn"
LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/server.log"
mkdir -p "$LOG_DIR"

# ── Cleanup ──────────────────────────────────────────────────
cleanup() {
    echo ""
    echo -e "${DIM}Shutting down...${NC}"
    [ -n "$TAIL_PID" ] && kill "$TAIL_PID" 2>/dev/null || true
    [ -n "$CAFFEINATE_PID" ] && kill "$CAFFEINATE_PID" 2>/dev/null || true
    stop_tunnel
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        sleep 1
        kill -0 "$SERVER_PID" 2>/dev/null && kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    # Final port check
    lsof -ti :"$PORT" | xargs kill -9 2>/dev/null || true
    echo -e "${DIM}Done.${NC}"
}
trap cleanup EXIT INT TERM

# ── Validate environment ─────────────────────────────────────
echo ""
echo -e "${BOLD}═══ Voice Frontend ═══${NC}"
echo ""

step "Environment"
echo ""

if [ ! -f "$VENV_UVICORN" ]; then
    fail "venv not found. Run: ./scripts/setup.sh"
    exit 1
fi
ok "Python venv"

# Check packages are importable
if "$VENV_PY" -c "import transport, engine_starter" 2>/dev/null; then
    ok "transport, engine_starter packages"
else
    fail "Packages not installed. Run: ./scripts/setup.sh"
    exit 1
fi

# Ollama check (non-fatal)
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    ok "Ollama running"
else
    warn "Ollama not running (LLM calls may fail)"
fi

# Optional: Twilio
if [ -n "${TWILIO_ACCOUNT_SID:-}" ]; then
    ok "Twilio credentials set"
else
    warn "Twilio not configured (using STUN only — localhost works fine)"
fi

# Optional: cloudflared
if command -v cloudflared >/dev/null 2>&1; then
    ok "cloudflared installed"
else
    warn "cloudflared not installed (only needed for --tunnel mode)"
fi

if [ "$CHECK_ONLY" = true ]; then
    echo ""
    ok "Validation complete (--check mode)"
    exit 0
fi

# ── Mode selection ───────────────────────────────────────────
if [ -z "$MODE" ]; then
    echo ""
    echo "  How do you want to connect?"
    echo ""
    echo "    1) Local      — http://localhost:$PORT (Mac browser)"
    echo "    2) LAN/WiFi   — https://<ip>:$PORT (phone on same WiFi)"
    echo "    3) Cellular   — Cloudflare Tunnel (phone on cell network)"
    echo ""
    read -rp "  Select mode [1/2/3]: " MODE_NUM

    case "$MODE_NUM" in
        1) MODE="local" ;;
        2) MODE="lan" ;;
        3) MODE="tunnel" ;;
        *) fail "Invalid selection: $MODE_NUM"; exit 1 ;;
    esac
fi

# ── Stale process cleanup ────────────────────────────────────
step "Starting ($MODE mode)"
echo ""

kill_stale_port "$PORT"

# ── caffeinate ───────────────────────────────────────────────
caffeinate -s -w $$ &
CAFFEINATE_PID=$!
ok "caffeinate enabled (Mac stays awake)"

# Lid-close warning
if ! pmset -g 2>/dev/null | grep -qi "sleepdisabled.*1"; then
    warn "Lid-close sleep is NOT disabled. Closing the lid kills the server."
    echo -e "    ${DIM}Run: sudo pmset disablesleep 1${NC}"
    echo -e "    ${DIM}Undo: sudo pmset disablesleep 0${NC}"
fi

# ── Mode-specific setup ──────────────────────────────────────
CONNECT_URL=""
UVICORN_HOST="127.0.0.1"
UVICORN_SSL_ARGS=""
CURL_FLAGS=""

case "$MODE" in
    local)
        CONNECT_URL="http://localhost:$PORT"
        ;;
    lan)
        LOCAL_IP=$(detect_local_ip)
        if [ -z "$LOCAL_IP" ]; then
            fail "Could not detect local IP. Are you connected to WiFi?"
            exit 1
        fi
        ok "Local IP: $LOCAL_IP"
        ensure_self_signed_cert "$REPO_ROOT"
        UVICORN_HOST="0.0.0.0"
        UVICORN_SSL_ARGS="--ssl-keyfile $REPO_ROOT/.ssl/key.pem --ssl-certfile $REPO_ROOT/.ssl/cert.pem"
        CONNECT_URL="https://$LOCAL_IP:$PORT"
        CURL_FLAGS="-k"  # skip cert verification for self-signed
        ;;
    tunnel)
        start_tunnel "$PORT" "$REPO_ROOT" || exit 1
        CONNECT_URL=$(get_tunnel_url)
        if [ -z "$CONNECT_URL" ]; then
            warn "Named tunnel has no DNS route configured."
            warn "Run setup_tunnel.sh and enter a subdomain, or use a quick tunnel."
            CONNECT_URL="http://localhost:$PORT"
        fi
        ok "Tunnel URL: $CONNECT_URL"
        ;;
esac

# ── Start uvicorn ────────────────────────────────────────────
echo ""
echo "  Starting uvicorn ($EXAMPLE_DIR)..."
echo "--- server start $(date '+%Y-%m-%d %H:%M:%S') ---" >> "$LOG_FILE"

cd "$REPO_ROOT/$EXAMPLE_DIR"

# shellcheck disable=SC2086
$VENV_UVICORN server:app \
    --host "$UVICORN_HOST" \
    --port "$PORT" \
    $UVICORN_SSL_ARGS \
    >> "$LOG_FILE" 2>&1 &
SERVER_PID=$!

cd "$REPO_ROOT"

# Wait for port
echo -n "  Waiting for port $PORT"
if ! wait_for_port "$PORT" 15; then
    echo ""
    fail "Server failed to bind port $PORT. Check: logs/server.log"
    exit 1
fi
echo ""

# Health check
HEALTH_BASE="http://localhost:$PORT"
if [ "$MODE" = "lan" ]; then
    HEALTH_BASE="https://localhost:$PORT"
fi

if wait_for_health "$HEALTH_BASE" 10 $CURL_FLAGS; then
    ok "Health check passed"
else
    warn "Health endpoint not responding (server may still work without /health route)"
fi

# ── Dashboard ────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${BOLD}Mode:${NC}  $MODE"
echo -e "  ${BOLD}URL:${NC}   ${CYAN}$CONNECT_URL${NC}"
echo -e "  ${BOLD}Admin:${NC} ${CYAN}$CONNECT_URL/admin${NC}"
echo -e "  ${BOLD}Logs:${NC}  logs/server.log"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# QR code for remote modes
if [ "$MODE" = "lan" ] || [ "$MODE" = "tunnel" ]; then
    show_qr "$CONNECT_URL"
fi

if [ "$MODE" = "lan" ]; then
    echo "  NOTE: Self-signed HTTPS for mic access (getUserMedia)."
    echo "  On first visit, Safari will show a certificate warning."
    echo "  Tap 'Show Details' → 'visit this website' → 'Visit Website'."
    echo ""
fi

# Local mode: open browser
if [ "$MODE" = "local" ]; then
    if command -v open >/dev/null 2>&1; then
        open "$CONNECT_URL"
        ok "Opened in browser"
    fi
fi

# ── Watchdog ─────────────────────────────────────────────────
CF_PID=$(get_tunnel_pid)
echo -e "  ${DIM}Watchdog active — health check every ${WATCHDOG_INTERVAL:-60}s (Ctrl+C to stop)${NC}"
echo ""

# Tail logs in foreground
tail -f "$LOG_FILE" &
TAIL_PID=$!

watchdog_loop "$HEALTH_BASE" "$SERVER_PID" "$CF_PID" $CURL_FLAGS || {
    echo "[watchdog] Restarting in 3s..."
    kill "$TAIL_PID" 2>/dev/null || true
    TAIL_PID=""
    kill "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    stop_tunnel
    sleep 3
    # Disarm trap before re-exec (cleanup already done above)
    trap - EXIT
    exec "$0" "--$MODE" --port "$PORT" --example "$EXAMPLE"
}
