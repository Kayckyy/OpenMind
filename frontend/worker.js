let ws = null;
let stopped = false;
const CHUNK_SIZE = 4096;

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'start') {
    stopped = false;
    const { wsUrl, params, audioData } = payload;

    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      ws.send(JSON.stringify(params));
      self.postMessage({ type: 'status', msg: 'Streaming...' });
      sendChunks(audioData);
    };

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        const msg = JSON.parse(event.data);
        self.postMessage({ type: 'status', msg: msg.status });
        return;
      }
      self.postMessage({ type: 'chunk', buffer: event.data }, [event.data]);
    };

    ws.onclose = () => self.postMessage({ type: 'done' });
    ws.onerror = () => self.postMessage({ type: 'error' });
  }

  if (type === 'start_youtube') {
    stopped = false;
    const { wsUrl, params } = payload;

    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
      ws.send(JSON.stringify(params));
      self.postMessage({ type: 'status', msg: 'Connecting...' });
    };

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        const msg = JSON.parse(event.data);
        self.postMessage({ type: 'status', msg: msg.status });
        return;
      }
      self.postMessage({ type: 'chunk', buffer: event.data }, [event.data]);
    };

    ws.onclose = () => self.postMessage({ type: 'done' });
    ws.onerror = () => self.postMessage({ type: 'error' });
  }

  if (type === 'stop') {
    stopped = true;
    if (ws) ws.close();
  }
};

async function sendChunks(audioData) {
  const chL = new Float32Array(audioData.left);
  const chR = new Float32Array(audioData.right);

  for (let i = 0; i < chL.length; i += CHUNK_SIZE) {
    if (stopped) break;

    const blockL = chL.slice(i, i + CHUNK_SIZE);
    const blockR = chR.slice(i, i + CHUNK_SIZE);

    const interleaved = new Float32Array(blockL.length + blockR.length);
    for (let j = 0; j < blockL.length; j++) {
      interleaved[j * 2] = blockL[j];
      interleaved[j * 2 + 1] = blockR[j];
    }

    ws.send(interleaved.buffer);
    await new Promise(r => setTimeout(r, 10));
  }

  if (!stopped) ws.send(new TextEncoder().encode('END'));
}
