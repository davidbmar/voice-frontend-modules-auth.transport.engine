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
