/**
 * WebSocket client for gaze events from the Python tracker.
 */

const DEFAULT_WS_URL = 'ws://127.0.0.1:8765';
const RECONNECT_MS = 2000;

export class GazeClient {
    /**
     * @param {string} url
     * @param {{ onGaze?: (gaze: string) => void, onStatus?: (status: string) => void }} callbacks
     */
    constructor(url = DEFAULT_WS_URL, { onGaze, onStatus } = {}) {
        this.url = url;
        this.onGaze = onGaze ?? (() => {});
        this.onStatus = onStatus ?? (() => {});
        this._socket = null;
        this._reconnectTimer = null;
        this._shouldReconnect = true;
    }

    connect() {
        this._shouldReconnect = true;
        this._open();
    }

    disconnect() {
        this._shouldReconnect = false;
        clearTimeout(this._reconnectTimer);
        if (this._socket) {
            this._socket.close();
            this._socket = null;
        }
        this.onStatus('disconnected');
    }

    _open() {
        if (this._socket) return;

        this.onStatus('connecting');
        const socket = new WebSocket(this.url);
        this._socket = socket;

        socket.addEventListener('open', () => {
            this.onStatus('connected');
        });

        socket.addEventListener('message', (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data && typeof data.gaze === 'string') {
                    this.onGaze(data.gaze);
                }
            } catch {
                // ignore malformed payloads
            }
        });

        socket.addEventListener('close', () => {
            this._socket = null;
            this.onStatus('disconnected');
            this._scheduleReconnect();
        });

        socket.addEventListener('error', () => {
            this.onStatus('error');
        });
    }

    _scheduleReconnect() {
        if (!this._shouldReconnect) return;
        clearTimeout(this._reconnectTimer);
        this._reconnectTimer = setTimeout(() => this._open(), RECONNECT_MS);
    }
}
