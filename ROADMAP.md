# Roadmap

## Boards as peers

Boards talk to each other directly, audio and video, over OmniPython's
`medialink`, and the server joins as one more peer through
`examples/medialink_peer.py`. It answers each turn by echo, by saying back
what Moonshine heard, or through /v1/voice when the chat model is loaded.

Examples to build on that, each using engines the server already has:

### Robot car

- **Stop for what's in the way.** The car streams its camera; detection and
  depth say what's ahead and how far, and the car stops and says why.
- **Follow me,** from pose or face tracking, and stop when a hand is raised.
- **Telepresence.** The videophone on wheels: you see and hear from the car
  and talk through it.
- **Patrol and listen.** The sound classifier hears a smoke alarm, glass
  breaking or a dog; the car says what and sends a clip to a desktop board.

### Robot

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
- **Knows who's asking,** from speaker recognition, so replies and notes are
  per person.
