#!/usr/bin/env bash
# One-time Cloudflare Tunnel setup.
# Creates a named tunnel and optionally routes a DNS subdomain.
# Run once, then use: ./scripts/run.sh --tunnel
set -euo pipefail

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
