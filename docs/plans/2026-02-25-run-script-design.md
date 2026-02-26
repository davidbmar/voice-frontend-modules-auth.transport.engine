# Run Script Design — 3-Mode Launcher

**Date:** 2026-02-25
**Status:** Approved

## Goal

Create a unified launcher script (`scripts/run.sh`) that supports three connectivity modes: Local, LAN/WiFi, and Cellular (Cloudflare Tunnel). This replaces the current manual `uvicorn server:app --port 8090` workflow.

## File Structure

```
scripts/
├── run.sh                  # Main entry point (~150 lines)
├── setup_tunnel.sh         # One-time named tunnel setup
└── lib/
    ├── colors.sh           # ok(), warn(), fail(), step(), dashboard formatting
    ├── network.sh          # detect_local_ip(), kill_stale_port(), wait_for_port()
    ├── cert.sh             # generate_self_signed_cert() for LAN HTTPS
    ├── health.sh           # check_health(), watchdog_loop()
    └── tunnel.sh           # start_tunnel(), stop_tunnel(), wait_for_tunnel_url()
```

## Modes

### Mode 1 — Local (`--local` or menu option 1)
- `uvicorn server:app --host 127.0.0.1 --port $PORT --reload`
- Opens `http://localhost:$PORT` in browser (macOS `open`)
- No cert, no tunnel

### Mode 2 — LAN/WiFi (`--lan` or menu option 2)
- Detect local IP via `ipconfig getifaddr en0`
- Generate self-signed cert to `.ssl/cert.pem` + `.ssl/key.pem` (reuse if exists)
- `uvicorn server:app --host 0.0.0.0 --port $PORT --ssl-keyfile --ssl-certfile`
- QR code with `https://<local-ip>:$PORT`
- Safari cert warning instructions

### Mode 3 — Cellular (`--tunnel` or menu option 3)
- `uvicorn server:app --host 127.0.0.1 --port $PORT`
- cloudflared (named tunnel from `.tunnel-config`, or quick tunnel)
- QR code with tunnel URL

## Common Features (all modes)

- **caffeinate** — prevent Mac sleep while running
- **Lid-close warning** — prompt user to `sudo pmset disablesleep 1`
- **Stale process cleanup** — kill any existing process on $PORT before starting
- **Health checks** — dot-progress indicator during startup
- **Colored dashboard** — mode, URL, admin URL, health status
- **Watchdog** — check health every 60s, auto-restart on failure

## CLI Interface

```
./scripts/run.sh                    # Interactive 3-mode menu
./scripts/run.sh --local            # Local mode, no interaction
./scripts/run.sh --lan              # LAN mode, no interaction
./scripts/run.sh --tunnel           # Cellular mode, no interaction
./scripts/run.sh --check            # Validate config only, don't start
./scripts/run.sh --port 9000        # Override port (default: 8090)
./scripts/run.sh --example minimal  # Run minimal-voice-app instead of with-admin
```

## Startup Flow

1. Parse flags
2. Validate environment (venv, uvicorn, packages, Ollama, optional Twilio/cloudflared)
3. If `--check` → print status, exit 0
4. If no mode flag → show interactive 3-mode menu
5. `kill_stale_port($PORT)`
6. caffeinate + lid-close warning
7. Mode-specific setup (cert generation / tunnel start)
8. Launch uvicorn in background
9. `wait_for_port()` + `check_health()`
10. Display colored dashboard
11. Watchdog loop (60s health checks, auto-restart)

## setup_tunnel.sh

One-time script to create a named Cloudflare tunnel:
- Authenticates with Cloudflare (if needed)
- Creates named tunnel
- Optionally routes a DNS subdomain
- Saves config to `.tunnel-config`

## Reference Projects

- `~/src/iphone-and-companion-transcribe-mode/scripts/run.sh` — 3-mode pattern, watchdog, QR codes
- `~/src/voice-calendar-scheduler-FSM/scripts/start.sh` — colored output, dashboard, validation
- `~/src/voice-calendar-scheduler-FSM/scripts/setup_tunnel.sh` — named tunnel setup
