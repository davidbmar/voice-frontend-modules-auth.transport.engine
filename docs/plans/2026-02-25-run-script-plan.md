# 3-Mode Run Script Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a modular `scripts/run.sh` that supports Local, LAN/WiFi, and Cellular modes for running the voice frontend server.

**Architecture:** Modular shell library (`scripts/lib/*.sh`) sourced by a main orchestrator (`scripts/run.sh`). Each lib file owns one concern (colors, networking, certs, health, tunnel). A separate `setup_tunnel.sh` handles one-time Cloudflare named tunnel setup.

**Tech Stack:** Bash, uvicorn, openssl (cert gen), cloudflared (tunnels), qrencode (QR codes)

**Design doc:** `docs/plans/2026-02-25-run-script-design.md`

**Reference scripts:**
- `~/src/iphone-and-companion-transcribe-mode/scripts/run.sh` — 3-mode pattern, watchdog, QR
- `~/src/voice-calendar-scheduler-FSM/scripts/start.sh` — colored output, dashboard
- `~/src/voice-calendar-scheduler-FSM/scripts/setup_tunnel.sh` — named tunnel setup

---

### Task 1: Add /health endpoint to with-admin example

The with-admin server.py has no `/health` endpoint. The run script needs one for health checks and watchdog.

**Files:**
- Modify: `examples/with-admin/server.py`
- Modify: `examples/minimal-voice-app/server.py`

**Step 1: Add health endpoint to with-admin**

In `examples/with-admin/server.py`, add after the `@app.get("/")` route:

```python
@app.get("/health")
async def health():
    return {"status": "ok"}
```

**Step 2: Add health endpoint to minimal-voice-app**

Same pattern in `examples/minimal-voice-app/server.py`.

**Step 3: Verify**

```bash
cd examples/with-admin && python -c "from server import app; print([r.path for r in app.routes])"
```

Expected: list includes `/health`

**Step 4: Commit**

```bash
git add examples/with-admin/server.py examples/minimal-voice-app/server.py
git commit -m "feat: add /health endpoint to example servers"
```

---

### Task 2: Create `scripts/lib/colors.sh`

**Files:**
- Create: `scripts/lib/colors.sh`

**Step 1: Write colors.sh**

```bash
#!/usr/bin/env bash
# Color codes and formatting helpers for run scripts.
# Source this file — do not execute directly.

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
step() { echo -e "\n${BOLD}── $1 ──${NC}"; }
```

**Step 2: Smoke test**

```bash
source scripts/lib/colors.sh && ok "test" && warn "test" && fail "test" && step "test"
```

Expected: colored output with ✓ ! ✗ symbols and a bold step header.

**Step 3: Commit**

```bash
git add scripts/lib/colors.sh
git commit -m "feat: add scripts/lib/colors.sh — colored output helpers"
```

---

### Task 3: Create `scripts/lib/network.sh`

**Files:**
- Create: `scripts/lib/network.sh`

**Step 1: Write network.sh**

```bash
#!/usr/bin/env bash
# Network utilities: IP detection, port management.
# Source this file — do not execute directly.

detect_local_ip() {
    # Returns the local WiFi IP address, or empty string on failure.
    local ip=""
    if command -v ipconfig >/dev/null 2>&1; then
        ip=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "")
    fi
    if [ -z "$ip" ] && command -v hostname >/dev/null 2>&1; then
        ip=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "")
    fi
    echo "$ip"
}

kill_stale_port() {
    # Kill any process listening on the given port.
    # Usage: kill_stale_port 8090
    local port="$1"
    local pids
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "$pids" | while read -r pid; do
            if [ "$pid" != "$$" ]; then
                kill "$pid" 2>/dev/null || true
            fi
        done
        sleep 1
        # Escalate if still alive
        pids=$(lsof -ti :"$port" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill -9 2>/dev/null || true
        fi
    fi
}

wait_for_port() {
    # Wait for a TCP listener on the given port. Returns 0 on success, 1 on timeout.
    # Usage: wait_for_port 8090 [timeout_seconds]
    local port="$1"
    local timeout="${2:-15}"
    for i in $(seq 1 "$timeout"); do
        if lsof -i :"$port" -sTCP:LISTEN >/dev/null 2>&1; then
            return 0
        fi
        echo -n "."
        sleep 1
    done
    return 1
}

show_qr() {
    # Display a QR code for the given URL (best effort).
    # Usage: show_qr "https://example.com"
    local url="$1"
    echo ""
    echo "  Scan this QR code on your phone:"
    echo ""
    if command -v qrencode >/dev/null 2>&1; then
        qrencode -t ANSIUTF8 "$url"
    elif python3 -c "import qrcode" 2>/dev/null; then
        python3 -c "
import qrcode
qr = qrcode.QRCode(border=1)
qr.add_data('$url')
qr.print_ascii(invert=True)
"
    else
        echo "  (no QR tool found — install: brew install qrencode)"
        echo ""
        echo "  Open manually: $url"
    fi
    echo ""
}
```

**Step 2: Smoke test**

```bash
source scripts/lib/network.sh && detect_local_ip
```

Expected: prints your local IP (e.g., `192.168.1.x`) or empty.

**Step 3: Commit**

```bash
git add scripts/lib/network.sh
git commit -m "feat: add scripts/lib/network.sh — IP detection, port mgmt, QR codes"
```

---

### Task 4: Create `scripts/lib/cert.sh`

**Files:**
- Create: `scripts/lib/cert.sh`

**Step 1: Write cert.sh**

```bash
#!/usr/bin/env bash
# Self-signed certificate generation for LAN HTTPS mode.
# Source this file — do not execute directly.

ensure_self_signed_cert() {
    # Generate a self-signed cert if one doesn't already exist.
    # Usage: ensure_self_signed_cert /path/to/project
    # Creates .ssl/cert.pem and .ssl/key.pem under the given directory.
    local project_root="$1"
    local ssl_dir="$project_root/.ssl"
    local cert="$ssl_dir/cert.pem"
    local key="$ssl_dir/key.pem"

    if [ -f "$cert" ] && [ -f "$key" ]; then
        echo "  Self-signed cert exists: $ssl_dir"
        return 0
    fi

    mkdir -p "$ssl_dir"

    local ip
    ip=$(detect_local_ip)
    local san="subjectAltName=IP:${ip:-127.0.0.1}"

    echo "  Generating self-signed certificate..."
    openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout "$key" \
        -out "$cert" \
        -days 365 \
        -subj "/CN=voice-frontend-dev" \
        -addext "$san" \
        2>/dev/null

    if [ -f "$cert" ] && [ -f "$key" ]; then
        echo "  Certificate created: $ssl_dir"
    else
        echo "  ERROR: Failed to generate certificate."
        return 1
    fi
}
```

**Step 2: Smoke test**

```bash
source scripts/lib/network.sh && source scripts/lib/cert.sh && ensure_self_signed_cert /tmp/test-cert && ls /tmp/test-cert/.ssl/ && rm -rf /tmp/test-cert/.ssl
```

Expected: prints "Generating self-signed certificate..." then lists `cert.pem  key.pem`.

**Step 3: Commit**

```bash
git add scripts/lib/cert.sh
git commit -m "feat: add scripts/lib/cert.sh — self-signed cert generation for LAN mode"
```

---

### Task 5: Create `scripts/lib/health.sh`

**Files:**
- Create: `scripts/lib/health.sh`

**Step 1: Write health.sh**

```bash
#!/usr/bin/env bash
# Health check and watchdog functions.
# Source this file — do not execute directly.

check_health() {
    # Hit the /health endpoint. Returns 0 if healthy, 1 otherwise.
    # Usage: check_health http://localhost:8090 [curl_flags]
    local base_url="$1"
    local extra_flags="${2:-}"
    curl -sf $extra_flags "${base_url}/health" >/dev/null 2>&1
}

wait_for_health() {
    # Wait for the health endpoint to respond. Prints dot progress.
    # Usage: wait_for_health http://localhost:8090 [timeout_seconds] [curl_flags]
    local base_url="$1"
    local timeout="${2:-15}"
    local extra_flags="${3:-}"

    echo -n "  Waiting for health"
    for i in $(seq 1 "$timeout"); do
        if check_health "$base_url" "$extra_flags"; then
            echo ""
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo ""
    return 1
}

watchdog_loop() {
    # Monitor health and break (for restart) on failure.
    # Usage: watchdog_loop http://localhost:8090 server_pid [curl_flags] [cf_pid]
    # Breaks out of the loop on failure so the caller can restart.
    local base_url="$1"
    local server_pid="$2"
    local curl_flags="${3:-}"
    local cf_pid="${4:-}"
    local interval="${WATCHDOG_INTERVAL:-60}"

    while true; do
        sleep "$interval"

        local reason=""

        # Check server process alive
        if [ -n "$server_pid" ] && ! kill -0 "$server_pid" 2>/dev/null; then
            reason="server process (PID $server_pid) died"
        fi

        # Check cloudflared alive (if applicable)
        if [ -z "$reason" ] && [ -n "$cf_pid" ] && ! kill -0 "$cf_pid" 2>/dev/null; then
            reason="cloudflared process (PID $cf_pid) died"
        fi

        # Check health endpoint
        if [ -z "$reason" ] && ! check_health "$base_url" "$curl_flags"; then
            reason="health endpoint unresponsive"
        fi

        if [ -n "$reason" ]; then
            echo ""
            echo "[watchdog] $(date '+%H:%M:%S') FAILURE: $reason"
            return 1
        fi
    done
}
```

**Step 2: Commit**

```bash
git add scripts/lib/health.sh
git commit -m "feat: add scripts/lib/health.sh — health check and watchdog"
```

---

### Task 6: Create `scripts/lib/tunnel.sh`

**Files:**
- Create: `scripts/lib/tunnel.sh`

**Step 1: Write tunnel.sh**

```bash
#!/usr/bin/env bash
# Cloudflare Tunnel management (named or quick tunnel).
# Source this file — do not execute directly.

# Module-level state
_CF_PID=""
_TUNNEL_URL=""
_TUNNEL_LOG=""

start_tunnel() {
    # Start a cloudflared tunnel. Sets _CF_PID and _TUNNEL_URL.
    # Usage: start_tunnel port project_root
    local port="$1"
    local project_root="$2"
    local config_file="$project_root/.tunnel-config"
    _TUNNEL_LOG="$project_root/logs/cloudflared.log"
    mkdir -p "$(dirname "$_TUNNEL_LOG")"

    if ! command -v cloudflared >/dev/null 2>&1; then
        fail "cloudflared not found. Install: brew install cloudflared"
        return 1
    fi

    if [ -f "$config_file" ]; then
        # Named tunnel
        local tunnel_name="" tunnel_url=""
        while IFS='=' read -r key value; do
            key=$(echo "$key" | tr -d "' ")
            value=$(echo "$value" | tr -d "' ")
            case "$key" in
                TUNNEL_NAME) tunnel_name="$value" ;;
                TUNNEL_URL)  tunnel_url="$value" ;;
            esac
        done < "$config_file"

        echo "  Using named tunnel: $tunnel_name"
        cloudflared tunnel --url "http://localhost:$port" run "$tunnel_name" > "$_TUNNEL_LOG" 2>&1 &
        _CF_PID=$!
        _TUNNEL_URL="$tunnel_url"

        # Wait for tunnel registration
        echo -n "  Waiting for tunnel"
        for i in $(seq 1 30); do
            if grep -q "Registered tunnel connection" "$_TUNNEL_LOG" 2>/dev/null; then
                echo ""
                return 0
            fi
            echo -n "."
            sleep 1
        done
        echo ""
        warn "Tunnel may not be fully connected (check logs/cloudflared.log)"
    else
        # Quick tunnel (random URL)
        echo "  No named tunnel found — using quick tunnel."
        echo "  Tip: run scripts/setup_tunnel.sh for a permanent URL."
        echo ""
        cloudflared tunnel --url "http://localhost:$port" > "$_TUNNEL_LOG" 2>&1 &
        _CF_PID=$!

        echo -n "  Waiting for tunnel URL"
        for i in $(seq 1 30); do
            _TUNNEL_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$_TUNNEL_LOG" 2>/dev/null | head -1 || echo "")
            if [ -n "$_TUNNEL_URL" ]; then
                echo ""
                return 0
            fi
            echo -n "."
            sleep 1
        done
        echo ""
        fail "Could not get tunnel URL after 30s. Check logs/cloudflared.log"
        return 1
    fi
}

stop_tunnel() {
    # Stop the cloudflared process.
    if [ -n "$_CF_PID" ] && kill -0 "$_CF_PID" 2>/dev/null; then
        kill "$_CF_PID" 2>/dev/null || true
        wait "$_CF_PID" 2>/dev/null || true
    fi
    _CF_PID=""
    _TUNNEL_URL=""
}

get_tunnel_url() { echo "$_TUNNEL_URL"; }
get_tunnel_pid() { echo "$_CF_PID"; }
```

**Step 2: Commit**

```bash
git add scripts/lib/tunnel.sh
git commit -m "feat: add scripts/lib/tunnel.sh — cloudflare tunnel management"
```

---

### Task 7: Create `scripts/run.sh` — the main orchestrator

This is the main script. It sources all lib files and orchestrates the full startup flow.

**Files:**
- Create: `scripts/run.sh`

**Step 1: Write run.sh**

```bash
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

# ── Parse flags ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)            MODE="local"; shift ;;
        --lan)              MODE="lan"; shift ;;
        --tunnel)           MODE="tunnel"; shift ;;
        --check)            CHECK_ONLY=true; shift ;;
        --port)             PORT="$2"; shift 2 ;;
        --example)          EXAMPLE="$2"; shift 2 ;;
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
CURL_FLAGS="-sf"

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
        CURL_FLAGS="-sfk"  # skip cert verification for self-signed
        ;;
    tunnel)
        start_tunnel "$PORT" "$REPO_ROOT" || exit 1
        CONNECT_URL=$(get_tunnel_url)
        ok "Tunnel URL: $CONNECT_URL"
        ;;
esac

# ── Start uvicorn ────────────────────────────────────────────
echo ""
echo "  Starting uvicorn ($EXAMPLE_DIR)..."
echo "--- server start $(date '+%Y-%m-%d %H:%M:%S') ---" >> "$LOG_FILE"

cd "$REPO_ROOT/$EXAMPLE_DIR"

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

if wait_for_health "$HEALTH_BASE" 10 "$CURL_FLAGS"; then
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

watchdog_loop "$HEALTH_BASE" "$SERVER_PID" "$CURL_FLAGS" "$CF_PID" || {
    echo "[watchdog] Restarting in 3s..."
    kill "$TAIL_PID" 2>/dev/null || true
    kill "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    stop_tunnel
    sleep 3
    # Re-exec ourselves with the same mode
    exec "$0" "--$MODE" --port "$PORT" --example "$EXAMPLE"
}
```

**Step 2: Make executable**

```bash
chmod +x scripts/run.sh
```

**Step 3: Smoke test (--check)**

```bash
./scripts/run.sh --check
```

Expected: colored validation output, then "Validation complete (--check mode)".

**Step 4: Commit**

```bash
git add scripts/run.sh
git commit -m "feat: add scripts/run.sh — 3-mode launcher (local/LAN/tunnel)"
```

---

### Task 8: Create `scripts/setup_tunnel.sh`

**Files:**
- Create: `scripts/setup_tunnel.sh`

**Step 1: Write setup_tunnel.sh**

Adapt from `~/src/voice-calendar-scheduler-FSM/scripts/setup_tunnel.sh` — same logic, adjusted project name.

```bash
#!/usr/bin/env bash
# One-time Cloudflare Tunnel setup.
# Creates a named tunnel and optionally routes a DNS subdomain.
# Run once, then use: ./scripts/run.sh --tunnel
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/.tunnel-config"

# Source colors
source "$PROJECT_ROOT/scripts/lib/colors.sh"

# ── Prerequisites ────────────────────────────────────────────
if ! command -v cloudflared >/dev/null 2>&1; then
    fail "cloudflared not found."
    echo "  Install: brew install cloudflared"
    exit 1
fi

if [ -f "$CONFIG_FILE" ]; then
    echo "Tunnel already configured: $CONFIG_FILE"
    cat "$CONFIG_FILE"
    echo ""
    read -rp "Reconfigure? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 0
fi

# ── Authenticate ─────────────────────────────────────────────
CERT="$HOME/.cloudflared/cert.pem"
if [ -f "$CERT" ]; then
    ok "Cloudflare auth found ($CERT)"
else
    echo "Opening browser to authenticate with Cloudflare..."
    cloudflared tunnel login
    if [ ! -f "$CERT" ]; then
        fail "Authentication failed — cert.pem not created."
        exit 1
    fi
fi

# ── Create tunnel ────────────────────────────────────────────
read -rp "Tunnel name [voice-frontend]: " TUNNEL_NAME
TUNNEL_NAME="${TUNNEL_NAME:-voice-frontend}"

if [[ "$TUNNEL_NAME" =~ [^a-zA-Z0-9_-] ]]; then
    fail "Tunnel name must contain only letters, numbers, hyphens, underscores."
    exit 1
fi

# Check if tunnel already exists
EXISTING_ID=$(TUNNEL_TARGET="$TUNNEL_NAME" cloudflared tunnel list --output json 2>/dev/null \
    | python3 -c "
import sys, json, os
raw = sys.stdin.read().strip()
if not raw:
    sys.exit(0)
tunnels = json.loads(raw)
target = os.environ['TUNNEL_TARGET']
for t in tunnels:
    if t['name'] == target:
        print(t['id'])
        break
" 2>/dev/null || true)

if [ -n "$EXISTING_ID" ]; then
    ok "Tunnel '$TUNNEL_NAME' already exists (ID: $EXISTING_ID)"
    TUNNEL_ID="$EXISTING_ID"
else
    echo "Creating tunnel '$TUNNEL_NAME'..."
    TUNNEL_ID=$(cloudflared tunnel create "$TUNNEL_NAME" 2>&1 \
        | grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' \
        | head -1)
    if [ -z "$TUNNEL_ID" ]; then
        fail "Failed to create tunnel."
        exit 1
    fi
    ok "Created tunnel: $TUNNEL_ID"
fi

# ── Optional DNS ─────────────────────────────────────────────
echo ""
echo "Optional: route a subdomain to this tunnel."
echo "  Example: voice.yourdomain.com"
echo "  (Leave blank to skip)"
read -rp "Subdomain (full hostname): " SUBDOMAIN

TUNNEL_URL=""
if [ -n "$SUBDOMAIN" ]; then
    echo "Routing $SUBDOMAIN → tunnel $TUNNEL_NAME..."
    cloudflared tunnel route dns "$TUNNEL_NAME" "$SUBDOMAIN" 2>&1 || true
    TUNNEL_URL="https://$SUBDOMAIN"
    ok "DNS configured: $TUNNEL_URL"
fi

# ── Save config ──────────────────────────────────────────────
cat > "$CONFIG_FILE" <<EOF
# Cloudflare Tunnel config — machine-specific, do not commit
TUNNEL_NAME='$TUNNEL_NAME'
TUNNEL_ID='$TUNNEL_ID'
TUNNEL_URL='$TUNNEL_URL'
EOF

echo ""
ok "Saved to $CONFIG_FILE"
echo ""
echo "  Start with tunnel: ./scripts/run.sh --tunnel"
```

**Step 2: Make executable**

```bash
chmod +x scripts/setup_tunnel.sh
```

**Step 3: Commit**

```bash
git add scripts/setup_tunnel.sh
git commit -m "feat: add scripts/setup_tunnel.sh — one-time named tunnel setup"
```

---

### Task 9: Update .gitignore

**Files:**
- Modify: `.gitignore`

**Step 1: Add .ssl and logs to .gitignore**

Append to `.gitignore`:

```
.ssl/
logs/
```

**Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore .ssl/ and logs/"
```

---

### Task 10: Make all lib scripts executable and set permissions

**Step 1: Set permissions**

```bash
chmod +x scripts/run.sh scripts/setup_tunnel.sh
chmod +x scripts/lib/*.sh
```

**Step 2: Integration test — local mode**

```bash
./scripts/run.sh --check
```

Expected: all validation checks print with colored ✓/! symbols, exits cleanly.

**Step 3: Commit all together**

```bash
git add -A scripts/
git commit -m "feat: complete 3-mode run script with modular lib"
```

---

### Task 11: Update setup.sh to mention run.sh

**Files:**
- Modify: `scripts/setup.sh`

**Step 1: Update the "Setup complete" message at the end**

Replace the current instructions at the bottom of `scripts/setup.sh`:

```bash
echo
echo "=== Setup complete ==="
echo
echo "To run:"
echo "  source .venv/bin/activate"
echo "  ./scripts/run.sh                # Interactive mode selection"
echo "  ./scripts/run.sh --local        # Local (localhost only)"
echo "  ./scripts/run.sh --lan          # LAN (phone on same WiFi)"
echo "  ./scripts/run.sh --tunnel       # Cellular (Cloudflare Tunnel)"
echo "  ./scripts/run.sh --check        # Validate config only"
echo
```

**Step 2: Commit**

```bash
git add scripts/setup.sh
git commit -m "docs: update setup.sh to reference run.sh"
```
