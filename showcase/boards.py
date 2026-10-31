"""
The board video: the endpoints a board calls on its own, without the chat,
each sent a test photo or clip from boards.html, with the request, the
reply and what it means on screen.

    python showcase/boards.py              # every scene
    python showcase/boards.py lines depth  # just these scenes, in order

Writes showcase/output/boards/boards.mp4 and its stills.
"""
import sys

from stage import IDLE_CUT, Stage, assemble

# Each scene, with the title card before it and how long its result stays on screen
SCENES = [
    ("detect", "Objects", "What's in front of the camera, and how far away", 4.0),
    ("pose", "Poses", "Where each person's joints are, and whether a hand is raised", 3.5),
    ("depth", "Depth", "How far away everything is, from one photo", 4.5),
    ("segment", "Finding things", "Anything with a name, and how far away it is", 4.0),
    ("lines", "Line following", "For a robot car: where the line is, and which way it turns", 4.0),
    ("actions", "Actions", "What people are doing, from photos sent one at a time", 3.0),
    ("ocr", "Reading", "The text in a photo, line by line", 3.5),
    ("quality", "Photo quality", "Which photo of a burst to keep", 3.5),
    ("classify", "Fast commands", "A voice command's intent in milliseconds, without the LLM", 4.0),
    ("rerank", "Reranking", "Which of a board's own passages answers a question", 4.0),
    ("sounds", "Sounds", "What's making a noise", 2.5),
    ("speakers", "Speakers", "Who spoke when", 2.5),
    ("pitch", "Tuning", "The note an instrument plays, and how far out of tune", 3.0),
    ("denoise", "Noise removal", "Clean speech from a noisy recording", 2.5),
]


def main(names) -> None:
    chosen = [scene for scene in SCENES if not names or scene[0] in names]
    with Stage("boards", "http://localhost:5000/showcase-pages/boards.html", ready="window.boardsReady") as s:
        s.card("lite-assistant for boards", "Endpoints a microcontroller or a robot calls with a photo or a recording", 4.5)
        for name, title, subtitle, hold in chosen:
            s.card(title, subtitle)
            started = s.now()
            try:
                s.page.evaluate("name => scenes[name]()", name)
            except Exception as e:
                s.note(f"error in {name}: {e}")
            # Waits for a reply, such as a model loading on first use, are cut short like the chat's
            for begun, ended in s.page.evaluate("waits.splice(0)"):
                if (ended - begun) / 1000 > IDLE_CUT:
                    s.idle.append((begun / 1000 - s.wall0, ended / 1000 - s.wall0))
            s.wait(0.5)
            s.still(name)
            s.wait(hold)
            s.note(f"{name} took {s.now() - started:.0f} s")
        s.card("lite-assistant", "github.com/PyDevices/lite-assistant", 5.0)
    assemble("boards")


if __name__ == "__main__":
    main(sys.argv[1:])
