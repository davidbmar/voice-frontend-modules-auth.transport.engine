#!/usr/bin/env bash
# Health check and watchdog functions.
# Source this file — do not execute directly.

check_health() {
    # Hit the /health endpoint. Returns 0 if healthy, 1 otherwise.
    # Usage: check_health http://localhost:8090 [extra_curl_flags...]
    local base_url="$1"
    shift
    curl -sf "$@" "${base_url}/health" >/dev/null 2>&1
}

wait_for_health() {
    # Wait for the health endpoint to respond. Prints dot progress.
    # Usage: wait_for_health http://localhost:8090 [timeout_seconds] [extra_curl_flags...]
    local base_url="$1"
    local timeout="${2:-15}"
    shift 2 2>/dev/null || shift

    echo -n "  Waiting for health"
    for i in $(seq 1 "$timeout"); do
        if check_health "$base_url" "$@"; then
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
    # Monitor health and return 1 on failure so the caller can restart.
    # Usage: watchdog_loop base_url server_pid cf_pid [extra_curl_flags...]
    local base_url="$1"
    local server_pid="$2"
    local cf_pid="${3:-}"
    shift 3 2>/dev/null || shift "$#"
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
        if [ -z "$reason" ] && ! check_health "$base_url" "$@"; then
            reason="health endpoint unresponsive"
        fi

        if [ -n "$reason" ]; then
            echo ""
            echo "[watchdog] $(date '+%H:%M:%S') FAILURE: $reason"
            return 1
        fi
    done
}
