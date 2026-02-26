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
        python3 - "$url" <<'PYEOF'
import sys, qrcode
qr = qrcode.QRCode(border=1)
qr.add_data(sys.argv[1])
qr.print_ascii(invert=True)
PYEOF
    else
        echo "  (no QR tool found — install: brew install qrencode)"
        echo ""
        echo "  Open manually: $url"
    fi
    echo ""
}
