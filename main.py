from typing import Union
import websockets
import asyncio
import wave
import numpy as np
from queue import Queue
import threading
import sounddevice as sd
from time import sleep
import soundfile as sf
import io

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from faster_whisper import WhisperModel

model_size = "tiny.en"

model = WhisperModel(model_size, device="cuda", compute_type="float16")

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

audio_queue = Queue()


def transcribe_audio():
    accumulated_audio = []
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            accumulated_audio.append(audio_data)

            # sd.play(audio_data, samplerate=44100)
            # sd.wait()  # Wait until audio finishes playing
            if len(accumulated_audio) > 40:
                combined_audio = np.concatenate(accumulated_audio)
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, combined_audio, 44100, format="WAV", subtype="FLOAT")
                wav_buffer.seek(0)
                # with wave.open("combined_audio.wav", "wb") as wf:
                #     wf.setnchannels(1)
                #     wf.setsampwidth(2)
                #     wf.setframerate(44100)
                #     wf.writeframes((combined_audio * 32767).astype(np.int16).tobytes())
                accumulated_audio = []
                # sf.write("combined.wav", combined_audio, 44100, subtype="FLOAT")
                # sd.play(combined_audio, samplerate=44100)
                segments, info = model.transcribe(
                    wav_buffer,
                    language="en",
                    # Source language (if known, or auto-detect)
                    beam_size=5,               # Larger beam size for better accuracy
                    temperature=0.0,           # Reduces randomness in output
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=True,
                    )

                print("Detected language '%s' with probability %f" %
                      (info.language, info.language_probability))

                for idx, segment in enumerate(segments):
                    print("[%.2fs -> %.2fs] %s" %
                          (segment.start, segment.end, segment.text))


transcription_thread = threading.Thread(target=transcribe_audio, daemon=True)
transcription_thread.start()


@ app.get("/")
def read_root():
    return {"Hello": "World"}


@ app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_frames = []
    i = 0

    try:
        while True:
            print("Receiving audio data...")
            data = await websocket.receive_bytes()
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            audio_queue.put(audio_float32)
            # sd.play(audio_float32)
            audio_frames.append(data)

            # audio_int16 = np.frombuffer(data, dtype=np.int16)

            # audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # segments, info = model.transcribe(
            #     audio_float32, beam_size=5, task="translate")

            # print("Detected language '%s' with probability %f" %
            #       (info.language, info.language_probability))

            # for idx, segment in enumerate(segments, start=1):
            #         print("[%.2fs -> %.2fs] %s" %
            #               (segment.start, segment.end, segment.text))
            # i += 1
    except websockets.exceptions.ConnectionClosedOK:
        pass
    finally:
        with wave.open("output.wav", "wb") as wf:
            wf.setnchannels(1)  # number of channels
            wf.setsampwidth(2)  # Set the sample width to 2 bytes
            wf.setframerate(44100)  # frame rate
            wf.writeframes(b''.join(audio_frames))

        await websocket.close()
