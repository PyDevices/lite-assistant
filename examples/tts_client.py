"""
Universal TTS Streaming Client
A zero-dependency socket-based client for interacting with the lite-assistant server.
Designed to be hardware-agnostic, running natively on both standard CPython 
and embedded MicroPython environments (like the ESP32) without modifications.
"""
import socket
import json

def get_available_voices(server_ip="127.0.0.1", port=5000):
    """
    Fetches the list of available voices from the lite-assistant server.
    
    Args:
        server_ip (str): IP address of the TTS server (default: "127.0.0.1")
        port (int): Port of the TTS server (default: 5000)
        
    Returns:
        list: A list of voice dictionary objects containing 'id' and 'name'.
    """
    s = socket.socket()
    try:
        addr = socket.getaddrinfo(server_ip, port)[0][-1]
        try:
            s.connect(addr)
        except OSError as e:
            raise ConnectionError(f"Could not connect to TTS server at {server_ip}:{port}. Is the server running?") from e
        
        request = (
            f"GET /voices HTTP/1.0\r\n"
            f"Host: {server_ip}\r\n"
            f"Connection: close\r\n\r\n"
        )
        s.send(request.encode())
        
        response = b""
        while True:
            chunk = s.recv(1024)
            if not chunk:
                break
            response += chunk
            
        parts = response.split(b"\r\n\r\n", 1)
        if len(parts) == 2:
            return json.loads(parts[1].decode('utf-8'))
    except Exception as e:
        print(f"Failed to fetch voices: {e}")
    finally:
        s.close()
    return []

def stream_tts(text, voice=None, model="kokoro", server_ip="127.0.0.1", port=5000, chunk_size=1024):
    """
    Generator that streams raw 16-bit PCM audio chunks from the TTS server.
    
    Args:
        text (str): The text to synthesize into speech.
        voice (str, optional): The voice ID to use. If None, auto-selects the first available.
        server_ip (str): IP address of the TTS server (default: "127.0.0.1")
        port (int): Port of the TTS server (default: 5000)
        chunk_size (int): Size of chunks to yield in bytes (default: 1024)
        
    Yields:
        bytes: Raw PCM audio data chunks.
    """
    # Auto-select voice if none is provided
    if not voice:
        voices = get_available_voices(server_ip, port)
        if voices:
            voice = voices[0]['id']
            print(f"Auto-selected voice: {voices[0]['name']}")
        else:
            voice = "af_heart" # fallback

    path = "/v1/audio/speech"
    payload = json.dumps({
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "pcm"
    })
    
    s = socket.socket()
    try:
        addr = socket.getaddrinfo(server_ip, port)[0][-1]
        try:
            s.connect(addr)
        except OSError as e:
            raise ConnectionError(f"Could not connect to TTS server at {server_ip}:{port}. Is the server running?") from e
        
        # HTTP/1.0, so the server streams the bare PCM and closes the
        # connection at the end, instead of framing it in chunked encoding
        request = (
            f"POST {path} HTTP/1.0\r\n"
            f"Host: {server_ip}\r\n"
            f"Authorization: Bearer dummy-token\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            f"Connection: close\r\n\r\n"
            f"{payload}"
        )
        s.send(request.encode())
        
        # 1. Strip standard HTTP response headers
        header_data = b""
        while b"\r\n\r\n" not in header_data:
            chunk = s.recv(1)
            if not chunk:
                break
            header_data += chunk
            
        # 2. Yield incoming socket chunks
        while True:
            packet = s.recv(chunk_size)
            if not packet:
                break
            yield packet
    finally:
        s.close()

def save_to_wav(filename, text, voice=None, model="kokoro", server_ip="127.0.0.1", port=5000, sample_rate=24000):
    """
    Synthesizes text and saves it directly to a local WAV file.
    Writes a standard 44-byte RIFF/WAVE header natively so it works on MicroPython.
    
    Args:
        filename (str): The output file path (e.g., "output.wav")
        text (str): The text to synthesize.
        voice (str, optional): The voice ID to use.
        server_ip (str): IP address of the TTS server (default: "127.0.0.1")
        port (int): Port of the TTS server (default: 5000)
        sample_rate (int): The sample rate of the PCM stream (default: 24000Hz)
        
    Returns:
        int: The total size of the PCM payload written in bytes.
    """
    import struct
    with open(filename, "wb") as f:
        # Write dummy header space
        f.write(b'\x00' * 44)
        
        data_size = 0
        for chunk in stream_tts(text, voice=voice, model=model, server_ip=server_ip, port=port):
            f.write(chunk)
            data_size += len(chunk)
            
        # Seek back to write actual RIFF WAV header
        f.seek(0)
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
    
    return data_size

if __name__ == "__main__":
    # Test generator
    print("Testing get_available_voices...")
    voices = get_available_voices("127.0.0.1")
    print(f"Found {len(voices)} voices.")
    
    print("Testing stream_tts generator...")
    total_bytes = 0
    for chunk in stream_tts("Hello from the universal TTS client!", server_ip="127.0.0.1"):
        total_bytes += len(chunk)
    print(f"Received {total_bytes} total bytes.")
    
    # Test saving to WAV
    wav_filename = "test_output.wav"
    print(f"\nTesting save_to_wav (generating '{wav_filename}')...")
    wav_bytes = save_to_wav(wav_filename, "This is a test of the wav file generation.", server_ip="127.0.0.1")
    print(f"Successfully saved {wav_bytes} bytes of PCM data to {wav_filename}!")
