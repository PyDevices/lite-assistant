"""
Universal STT Client SDK
Provides a robust, zero-dependency HTTP client for sending audio to the local
STT endpoint for transcription via multipart/form-data.
"""
import urllib.request
import os

def transcribe_audio_bytes(audio_bytes: bytes, server_url="http://localhost:5000/v1/audio/transcriptions"):
    """
    Sends an in-memory audio bytes object to the local STT endpoint and returns the transcription.
    """
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n')
    body.extend(b'Content-Type: audio/wav\r\n\r\n')
    body.extend(audio_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    
    req = urllib.request.Request(server_url, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    req.add_header('Content-Length', len(body))
    
    try:
        with urllib.request.urlopen(req) as response:
            result = response.read().decode('utf-8')
            return result
    except urllib.error.URLError as e:
        return f"Error making request: {e}"

def transcribe_audio_file(file_path: str, server_url="http://localhost:5000/v1/audio/transcriptions"):
    """
    Reads a local file into memory and uses transcribe_audio_bytes to transcribe it.
    """
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
        
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    return transcribe_audio_bytes(file_bytes, server_url)

if __name__ == "__main__":
    # Example usage using the test_audio.wav
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(script_dir, "..", "tests", "test_audio.wav")
    print(f"Attempting to transcribe file {test_file}...")
    
    result = transcribe_audio_file(test_file)
    print("Transcription Result:")
    print(result)
