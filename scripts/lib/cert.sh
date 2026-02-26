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

    local ip
    ip=$(detect_local_ip)

    if [ -f "$cert" ] && [ -f "$key" ]; then
        local cert_ip
        cert_ip=$(openssl x509 -noout -ext subjectAltName -in "$cert" 2>/dev/null \
            | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "")
        if [ "$cert_ip" = "${ip:-127.0.0.1}" ]; then
            echo "  Self-signed cert exists and IP matches: $ssl_dir"
            return 0
        fi
        echo "  IP changed ($cert_ip -> $ip) — regenerating cert..."
    fi

    mkdir -p "$ssl_dir"
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
