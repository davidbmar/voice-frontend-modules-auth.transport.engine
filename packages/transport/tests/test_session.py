"""Tests for transport.session — WebRTC session lifecycle."""

import re
import pytest

from transport.session import WebRTCSession, ice_servers_to_rtc


class TestCleanForSpeech:
    def test_strips_markdown_headers(self):
        assert WebRTCSession._clean_for_speech("## Hello World") == "Hello World"

    def test_strips_bold_and_italic(self):
        result = WebRTCSession._clean_for_speech("**bold** and *italic*")
        assert "bold" in result
        assert "*" not in result

    def test_strips_bullet_points(self):
        text = "- Item one\n- Item two"
        result = WebRTCSession._clean_for_speech(text)
        assert result.startswith("Item one")
        assert "-" not in result

    def test_strips_urls(self):
        text = "Visit https://example.com for more"
        result = WebRTCSession._clean_for_speech(text)
        assert "https://" not in result

    def test_strips_inline_code(self):
        result = WebRTCSession._clean_for_speech("Run `npm install` to start")
        assert "`" not in result
        assert "npm install" in result

    def test_strips_markdown_links(self):
        result = WebRTCSession._clean_for_speech("See [docs](https://example.com) for help")
        assert "docs" in result
        assert "[" not in result
        assert "https://" not in result


class TestSplitSentences:
    def test_splits_on_period(self):
        result = WebRTCSession._split_sentences("Hello. World.")
        assert result == ["Hello.", "World."]

    def test_single_sentence(self):
        result = WebRTCSession._split_sentences("Just one sentence.")
        assert result == ["Just one sentence."]

    def test_empty_string(self):
        result = WebRTCSession._split_sentences("")
        assert result == []

    def test_splits_on_exclamation_and_question(self):
        result = WebRTCSession._split_sentences("Hello! How are you? Fine.")
        assert len(result) == 3


class TestIceServersToRtc:
    def test_converts_urls_string(self):
        servers = [{"urls": "stun:stun.example.com:3478"}]
        result = ice_servers_to_rtc(servers)
        assert len(result) == 1

    def test_handles_url_key(self):
        servers = [{"url": "turn:turn.example.com:3478", "username": "u", "credential": "c"}]
        result = ice_servers_to_rtc(servers)
        assert len(result) == 1

    def test_empty_list(self):
        result = ice_servers_to_rtc([])
        assert result == []
