from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import wave
import numpy as np
from queue import Queue
import threading
import soundfile as sf
import io

from whisper_streaming.backends import FasterWhisperASR
from whisper_streaming.whisper_online import online_factory

SAMPLE_RATE = 16000
SAMPLES_PER_SEC = SAMPLE_RATE * 1
BYTES_PER_SAMPLE = 2  # s16le = 2 bytes per sample
BYTES_PER_SEC = SAMPLES_PER_SEC * BYTES_PER_SAMPLE

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Example dictionary
args_dict = {
    "buffer_trimming": "segment",
    "buffer_trimming_sec": 15,
    "min_chunk_size": 1.0,
    "model": "medium",
    "model_cache_dir": None,
    "model_dir": None,
    "lan": "en",
    "task": "transcribe",
    "backend": "faster-whisper",
    "vac": False,
    "vac_chunk_size": 0.04,
    "vad": False,
    "log_level": "DEBUG"
}

# Convert dictionary to object
args = Args(**args_dict)

asr = FasterWhisperASR(modelsize='medium', lan='auto')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

audio_queue = Queue()


def transcribe_audio():
    online = online_factory(args, asr, None)
    chunk_history = []
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            # wav_buffer = io.BytesIO()
            # sf.write(wav_buffer, audio_data, 44100,
            #          format="WAV", subtype="FLOAT")
            # wav_buffer.seek(0)
            # print("audio_data", audio_data)
            online.insert_audio_chunk(audio_data)
            beg_trans, end_trans, trans = online.process_iter()
            if trans:
                chunk_history.append({
                    "beg": beg_trans,
                    "end": end_trans,
                    "text": trans,
                    "speaker": "0"
                })
            buffer = online.concatenate_tsw(online.transcript_buffer.buffer)[2]

            lines = [
                {
                    "speaker": "0",
                    "text": "",
                }
            ]
            for ch in chunk_history:
                lines[-1]["text"] += ch['text']

            response = {"lines": lines, "buffer": buffer}
            print("Response", response)


# transcription_thread = threading.Thread(target=transcribe_audio, daemon=True)
# transcription_thread.start()


@ app.get("/")
def read_root():
    return {"Hello": "World"}


@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection opened.")
    audio_frames = []

    transcription_thread = threading.Thread(
        target=transcribe_audio, daemon=True)
    transcription_thread.start()

    try:
        while True:
            print("Receiving audio data...")
            data = await websocket.receive_bytes()
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_queue.put(audio_float32)
            # sd.play(audio_float32)
            audio_frames.append(data)
    except websocket.exceptions.ConnectionClosedOK:
        pass
    finally:
        with wave.open("output.wav", "wb") as wf:
            wf.setnchannels(1)  # number of channels
            wf.setsampwidth(2)  # Set the sample width to 2 bytes
            wf.setframerate(44100)  # frame rate
            wf.writeframes(b''.join(audio_frames))

        await websocket.close()
