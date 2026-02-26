/**
 * VoiceWebRTCClient — framework-agnostic browser WebRTC client.
 *
 * Connects to a transport.signaling server, negotiates WebRTC,
 * and streams bidirectional audio.
 *
 * Usage:
 *   import { VoiceWebRTCClient } from './voice-webrtc-client.js';
 *
 *   const client = new VoiceWebRTCClient({ signalingUrl: '/ws' });
 *   client.on('ready', () => console.log('ICE servers received'));
 *   client.on('connected', () => console.log('Call active'));
 *   client.on('failed', (reason) => console.log('Failed:', reason));
 *   client.on('ended', () => console.log('Call ended'));
 *   client.on('log', (msg, level) => console.log(`[${level}] ${msg}`));
 *
 *   document.getElementById('call-btn').onclick = () => {
 *       if (client.inCall) client.hangUp();
 *       else client.startCall();
 *   };
 */

export class VoiceWebRTCClient {
    constructor(options = {}) {
        this.signalingUrl = options.signalingUrl || '/ws';
        this.audio = Object.assign({
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            sampleRate: 48000,
        }, options.audio || {});
        this.iceGatheringTimeout = options.iceGatheringTimeout || 10000;

        this._ws = null;
        this._pc = null;
        this._localStream = null;
        this._iceServers = [];
        this._listeners = {};
        this.inCall = false;

        this._keepaliveId = null;
        this._connectSignaling();
    }

    // -- Event emitter ---------------------------------------------

    on(event, fn) {
        if (!this._listeners[event]) this._listeners[event] = [];
        this._listeners[event].push(fn);
    }

    _emit(event, ...args) {
        (this._listeners[event] || []).forEach(fn => fn(...args));
    }

    // -- Signaling -------------------------------------------------

    _connectSignaling() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = this.signalingUrl.startsWith('ws')
            ? this.signalingUrl
            : proto + '//' + location.host + this.signalingUrl;

        this._emit('log', 'Connecting to ' + url, 'info');
        this._ws = new WebSocket(url);

        this._ws.onopen = () => {
            this._emit('log', 'Signaling connected', 'info');
            this._ws.send(JSON.stringify({ type: 'hello' }));
            this._startKeepalive();
        };

        this._ws.onmessage = (event) => {
            let msg;
            try { msg = JSON.parse(event.data); }
            catch { this._emit('log', 'Invalid JSON from server', 'error'); return; }

            switch (msg.type) {
                case 'hello_ack':
                    this._iceServers = msg.ice_servers || [];
                    this._emit('log', 'Got ' + this._iceServers.length + ' ICE server(s)', 'info');
                    this._emit('ready');
                    break;
                case 'webrtc_answer':
                    this._handleAnswer(msg.sdp);
                    break;
                case 'error':
                    this._emit('log', 'Server error: ' + msg.message, 'error');
                    if (this.inCall) this._cleanup();
                    break;
                case 'pong':
                    break;
                default:
                    this._emit('log', 'Unknown message: ' + msg.type, 'info');
            }
        };

        this._ws.onclose = () => {
            this._emit('log', 'Signaling disconnected', 'info');
            this._emit('ended');
            this._stopKeepalive();
            this._ws = null;
            if (this.inCall) this._cleanup();
        };

        this._ws.onerror = () => {
            this._emit('log', 'WebSocket error', 'error');
        };
    }

    _startKeepalive() {
        this._keepaliveId = setInterval(() => {
            if (this._ws && this._ws.readyState === WebSocket.OPEN) {
                this._ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }

    _stopKeepalive() {
        if (this._keepaliveId) {
            clearInterval(this._keepaliveId);
            this._keepaliveId = null;
        }
    }

    // -- WebRTC ----------------------------------------------------

    async startCall() {
        if (this.inCall) return;

        try {
            this._localStream = await navigator.mediaDevices.getUserMedia({
                audio: this.audio,
                video: false,
            });
        } catch (err) {
            this._emit('failed', 'Microphone access denied: ' + err.message);
            return;
        }

        const rtcConfig = {
            iceServers: this._iceServers.map(s => {
                const entry = { urls: s.urls || s.url || '' };
                if (s.username) entry.username = s.username;
                if (s.credential) entry.credential = s.credential;
                return entry;
            }),
        };

        this._pc = new RTCPeerConnection(rtcConfig);

        this._localStream.getTracks().forEach(t => this._pc.addTrack(t, this._localStream));

        this._pc.ontrack = (event) => {
            this._emit('log', 'Remote audio track received', 'info');
            const audio = document.createElement('audio');
            audio.autoplay = true;
            audio.playsInline = true;
            audio.srcObject = event.streams[0] || new MediaStream([event.track]);
            document.body.appendChild(audio);
        };

        this._pc.oniceconnectionstatechange = () => {
            const state = this._pc.iceConnectionState;
            this._emit('log', 'ICE state: ' + state, 'info');
            if (state === 'connected') this._emit('connected');
            else if (state === 'disconnected' || state === 'failed') {
                this._emit('failed', 'ICE ' + state);
                this.hangUp();
            }
        };

        try {
            const offer = await this._pc.createOffer();
            await this._pc.setLocalDescription(offer);

            await new Promise((resolve) => {
                if (this._pc.iceGatheringState === 'complete') { resolve(); return; }
                const timer = setTimeout(() => {
                    this._emit('log', 'ICE gathering timed out, proceeding with partial candidates', 'error');
                    resolve();
                }, this.iceGatheringTimeout);
                this._pc.onicegatheringstatechange = () => {
                    if (this._pc.iceGatheringState === 'complete') {
                        clearTimeout(timer);
                        resolve();
                    }
                };
            });

            this._ws.send(JSON.stringify({
                type: 'webrtc_offer',
                sdp: this._pc.localDescription.sdp,
            }));

            this.inCall = true;
        } catch (err) {
            this._emit('failed', 'Offer creation failed: ' + err.message);
            this._cleanup();
        }
    }

    async _handleAnswer(sdp) {
        if (!this._pc) return;
        try {
            await this._pc.setRemoteDescription(new RTCSessionDescription({ type: 'answer', sdp }));
        } catch (err) {
            this._emit('log', 'Failed to set answer: ' + err.message, 'error');
        }
    }

    hangUp() {
        if (this._ws && this._ws.readyState === WebSocket.OPEN) {
            this._ws.send(JSON.stringify({ type: 'hangup' }));
        }
        this._cleanup();
        this._emit('ended');
    }

    _cleanup() {
        this.inCall = false;
        if (this._pc) { this._pc.close(); this._pc = null; }
        if (this._localStream) {
            this._localStream.getTracks().forEach(t => t.stop());
            this._localStream = null;
        }
    }

    destroy() {
        this.hangUp();
        this._stopKeepalive();
        if (this._ws) { this._ws.close(); this._ws = null; }
    }
}
