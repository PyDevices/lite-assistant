# Remove noise from speech

`POST /v1/audio/denoise` takes speech, as `/v1/audio/classifications` does, up to 64 s, and returns it as a 16 kHz WAV with the background noise taken out by CMGAN. On speech with white noise at 5 dB, it raised the signal-to-noise ratio from 4.9 dB to 12.7. It takes about as long as the audio.

```bash
curl -F file=@noisy.wav http://localhost:5000/v1/audio/denoise -o clean.wav
```
