#!/usr/bin/env python3
"""
Record audio from the default microphone (short clip) and POST to /api/transcribe,
then send the transcription to /api/chat. Requires `sounddevice` and `soundfile`.
"""
import sys
import argparse
import sounddevice as sd
import soundfile as sf
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=float, default=5.0)
parser.add_argument("--url", default="http://localhost:5000")
args = parser.parse_args()

def record_to_wav(duration, filename, samplerate=16000):
    print(f"Recording {duration}s...")
    data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, data, samplerate)


def main():
    import tempfile
    tf = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    fname = tf.name
    tf.close()
    record_to_wav(args.duration, fname)
    print("Uploading to server for transcription...")
    files = {'file': open(fname, 'rb')}
    r = requests.post(args.url + '/api/transcribe', files=files)
    if r.status_code != 200:
        print('Transcription failed:', r.text)
        sys.exit(1)
    text = r.json().get('text', '')
    print('Transcribed:', text)
    # send to chat
    r2 = requests.post(args.url + '/api/chat', json={'message': text, 'dry_run': True})
    print('AI reply:', r2.json())

if __name__ == '__main__':
    main()
