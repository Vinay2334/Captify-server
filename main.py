from typing import Union
import websockets
import asyncio
import wave

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from faster_whisper import WhisperModel

model_size = "medium"

model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

app = FastAPI()

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_frames = []
    i = 0

    try:
        while True:
            print("Receiving audio data...")
            data = await websocket.receive_bytes()
            audio_frames.append(data)
            
            # Save each chunk to a separate file
            with wave.open(f"audio_chunk_{i}.wav", "wb") as chunk_file:
                chunk_file.setnchannels(1)  # number of channels
                chunk_file.setsampwidth(2)  # Set the sample width to 2 bytes
                chunk_file.setframerate(44100)  # frame rate
                chunk_file.writeframes(data)
            i += 1
    except websockets.exceptions.ConnectionClosedOK:
        pass
    finally:
        with wave.open("output.wav", "wb") as wf:
            wf.setnchannels(1)  # number of channels
            wf.setsampwidth(2)  # Set the sample width to 2 bytes
            wf.setframerate(44100)  # frame rate
            wf.writeframes(b''.join(audio_frames))

        await websocket.close()
