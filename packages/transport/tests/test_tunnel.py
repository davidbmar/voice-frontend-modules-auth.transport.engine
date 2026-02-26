"""Tests for transport.tunnel -- Cloudflare tunnel management."""

import pytest

from transport.tunnel import CloudflareTunnel


class TestCloudflareTunnel:
    def test_init_defaults(self):
        t = CloudflareTunnel(local_port=8090)
        assert t.local_port == 8090
        assert t.url is None

    def test_init_with_config(self, tmp_path):
        config = tmp_path / ".tunnel-config"
        config.write_text(
            "TUNNEL_NAME=test-tunnel\nTUNNEL_ID=abc123\nTUNNEL_URL=https://test.example.com\n"
        )
        t = CloudflareTunnel(local_port=8090, config_path=str(config))
        assert t._tunnel_name == "test-tunnel"
        assert t._tunnel_url == "https://test.example.com"

    def test_missing_config_uses_quick_tunnel(self, tmp_path):
        config = tmp_path / ".tunnel-config-missing"
        t = CloudflareTunnel(local_port=8090, config_path=str(config))
        assert t._tunnel_name == ""

    def test_config_with_comments(self, tmp_path):
        config = tmp_path / ".tunnel-config"
        config.write_text("# comment\nTUNNEL_NAME=my-tunnel\n")
        t = CloudflareTunnel(local_port=8090, config_path=str(config))
        assert t._tunnel_name == "my-tunnel"
