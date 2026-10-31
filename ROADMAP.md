# Roadmap

## Boards as peers

Boards will talk to each other directly, audio and video, half duplex, over a
PyDevices package whose working name is `medialink`. The server then joins as one more
peer that speaks the same protocol. The server joins today through
`examples/medialink_peer.py`, which answers each turn by echo, by saying back
what Moonshine heard, or through /v1/voice when the chat model is loaded.

Examples to build once the server is a peer, each using engines the server
already has:

### Robot car

- **Talk to drive it.** Push-to-talk from a phone or a board; Moonshine
  transcribes and the fast classifier turns "left", "faster", "stop" into a
  command in milliseconds, without the LLM.
- **Stop for what's in the way.** The car streams its camera; detection and
  depth say what's ahead and how far, and the car stops and says why.
- **Follow a line,** from `/v1/lines` on the streamed frames.
- **Follow me,** from pose or face tracking, and stop when a hand is raised.
- **Telepresence.** The videophone on wheels: you see and hear from the car
  and talk through it.
- **Patrol and listen.** The sound classifier hears a smoke alarm, glass
  breaking or a dog; the car says what and sends a clip to a desktop board.

### Robot

- **Find a thing by name.** "Find the red cup": segmentation by name plus
  depth gives where it is and how far.
- **Narrate what it sees,** on request, from the LLM looking at a frame.
- **Gestures as commands:** a raised hand stops it; a wave starts it, from
  the action model.

### Desktop assistant

- **Wake word, then a conversation** over the realtime API, with replies
  spoken as each sentence is written.
- **Reminders and timers it speaks unasked,** pushed by the server.
- **Announcements to every board,** or to one by name: "dinner's ready".
- **Who's at the door.** The video doorbell's frames go to face detection
  and speaker recognition, and the desktop board says who it is.
- **Baby monitor alerts.** The sound classifier hears crying on the monitor's
  stream and the desktop board says so.
- **Read it to me.** Hold a page up to the camera; OCR reads it aloud.
- **Knows who's asking,** from speaker recognition, so replies and notes are
  per person.
