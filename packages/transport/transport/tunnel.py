"""Cloudflare Tunnel management -- named tunnels or quick (random URL) tunnels.

Wraps the `cloudflared` CLI binary as a subprocess.
"""

import logging
import re
import signal
import subprocess
import time
from pathlib import Path

log = logging.getLogger("transport.tunnel")


class CloudflareTunnel:
    """Manage a Cloudflare tunnel for exposing a local server.

    Uses a named tunnel if .tunnel-config exists (from setup_tunnel.sh),
    otherwise falls back to a quick tunnel with a random *.trycloudflare.com URL.
    """

    def __init__(self, local_port: int, config_path: str = ".tunnel-config"):
        self.local_port = local_port
        self.url: str | None = None
        self._process: subprocess.Popen | None = None
        self._tunnel_name = ""
        self._tunnel_id = ""
        self._tunnel_url = ""

        config = Path(config_path)
        if config.exists():
            self._load_config(config)

    def _load_config(self, path: Path):
        """Parse KEY=VALUE config file."""
        for line in path.read_text().splitlines():
            line = line.strip()
            if "=" not in line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key == "TUNNEL_NAME":
                self._tunnel_name = value
            elif key == "TUNNEL_ID":
                self._tunnel_id = value
            elif key == "TUNNEL_URL":
                self._tunnel_url = value

    def start(self):
        """Start the cloudflared tunnel subprocess."""
        if self._tunnel_name:
            cmd = [
                "cloudflared",
                "tunnel",
                "--url",
                f"http://localhost:{self.local_port}",
                "run",
                self._tunnel_name,
            ]
            self.url = self._tunnel_url
            log.info("Starting named tunnel: %s -> %s", self._tunnel_name, self.url)
        else:
            cmd = [
                "cloudflared",
                "tunnel",
                "--url",
                f"http://localhost:{self.local_port}",
            ]
            log.info("Starting quick tunnel on port %d", self.local_port)

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        if not self._tunnel_name:
            self._wait_for_url()

    def _wait_for_url(self, timeout: float = 30.0):
        """Parse the quick tunnel URL from cloudflared output."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self._process is None or self._process.poll() is not None:
                break
            line = self._process.stdout.readline()
            if not line:
                continue
            match = re.search(r"(https://\S+\.trycloudflare\.com)", line)
            if match:
                self.url = match.group(1)
                log.info("Quick tunnel URL: %s", self.url)
                return
        log.warning("Could not determine tunnel URL within %.0fs", timeout)

    def stop(self):
        """Stop the cloudflared tunnel."""
        if self._process:
            log.info("Stopping tunnel")
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            self.url = None
