import pyaudio
import wave
import numpy as np
from faster_whisper import WhisperModel
import os

NEON_GREEN = "\033[92m"  # Bright Green
RESET_COLOR = "\033[0m"

def transcribe_chunk(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = " ".join(seg.text for seg in segments)
    return transcription

def record_chunk(p, stream, file_path, chunk_length=1):
    frames=[]
    for _ in range(0, int(16000/1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def main():
    model_size="tiny.en"
    model= WhisperModel(model_size, device="cuda", compute_type="float16")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)
    accumulated_transcription=""

    try:
        while True:
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file)
            transcription = transcribe_chunk(model, chunk_file)
            print(NEON_GREEN+transcription+RESET_COLOR)
            os.remove(chunk_file)
            accumulated_transcription += transcription+" "
    except KeyboardInterrupt:
        print("Stopping...")
        with open("log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)
    finally:
        print("LOG:"+accumulated_transcription)
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
