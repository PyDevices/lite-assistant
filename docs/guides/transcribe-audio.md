# Transcribe audio

```bash
curl http://localhost:5000/v1/audio/transcriptions -F file="@tests/test_audio.wav"
```

`model=whisper` uses Whisper large-v3-turbo instead of Moonshine, and `language` (`fr`, `es` and so on) says what's spoken; without it, Whisper works it out. It's slower: 4.3 s for a 2 s Spanish question and 10.7 s for a 21 s conversation, against Moonshine's 0.3 s for 6 s. It's more accurate, though: it got every word of a clip in white noise as loud as the speech, where Moonshine got two wrong, and of a French request Moonshine turned into nonsense.

`denoise=true` takes the background noise out with CMGAN first. It's slow, about as long as the audio, and it helps only so far: with white noise as loud as the speech, it fixed one of Moonshine's two wrong words, but with the noise twice as loud it made the transcript worse. Moonshine copes with moderate noise by itself.
