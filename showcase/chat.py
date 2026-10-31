"""
The chat showcase: one long video of the chat page, and stills, showing
what it can do, with Gemma 4 E4B answering and the replies spoken.

    python showcase/chat.py            # every scene
    python showcase/chat.py plant docs # just these scenes, in order

Writes showcase/output/chat/chat.mp4 and its stills. Each scene starts a
new chat, so the model's context never fills up. A scene that fails is
noted in the log and the rest still run.
"""
import sys
import time

from stage import Stage, assemble

SETTINGS = {"model": "gemma-4-E4B-it", "speak": True, "suggest": True, "think": False}


def new_chat(s: Stage) -> None:
    s.click("#newchat")
    s.wait(0.8)


def intro(s: Stage) -> None:
    s.card("lite-assistant", "A chat page and an OpenAI-style API for small models running on this PC's CPU", 4.5)


def clock(s: Stage) -> None:
    s.card("Tools", "The model calls tools on its own when a question needs one")
    s.ask("What time is it?", still="time")


def seeing(s: Stage) -> None:
    s.card("Seeing", "Photos go to the model with the question; detectors count and measure for it")
    s.ask("What's going on in this photo?", "photos/000000001000.jpg", still="photo")
    s.ask("How many people are there, and who is smiling?", still="people")


def dog_distance(s: Stage) -> None:
    s.card("Depth", "How far away things are, from one photo")
    s.ask("How far away is the dog?", "dog.png", still="dog")


def editing(s: Stage) -> None:
    s.card("Editing photos", "Brighten, remove backgrounds, erase things and upscale, one step after another")
    s.ask("This photo is too dark. Brighten it.", "dark.jpg")
    s.ask("Now upscale it.", still="brighten")
    new_chat(s)
    s.ask("Remove the background.", "edit/beagle.jpg")
    s.ask("Now upscale that.", still="beagle")
    new_chat(s)
    s.ask("Erase the glasses from the table.", "showcase/glasses.jpg", still="erase")


def reading(s: Stage) -> None:
    s.card("Reading", "Text in photos, read and translated")
    s.ask("What does this sign say?", "sign.jpg")
    s.ask("Translate it into Spanish.", still="sign")


def plant(s: Stage) -> None:
    s.card("Plants", "A classifier trained on 1,081 species names the plant; the model tells you about it")
    s.ask("What plant is this?", "showcase/poppy.jpg", still="plant")


def documents(s: Stage) -> None:
    s.card("Documents", "PDFs, Word files and text files are read into the chat")
    s.ask("How do I clean the filter?", "documents/dishwasher_manual.pdf", still="pdf")
    new_chat(s)
    s.ask("What does my neighbour want, in one sentence?", "documents/letter_from_neighbour.docx", still="docx")


def sounds(s: Stage) -> None:
    s.card("Hearing", "Sounds, speakers and notes, from audio files")
    s.ask("What's that sound?", "sounds/alarm.wav", still="sounds")
    new_chat(s)
    s.ask("Who is speaking in this recording, and when?", "audio/mm/talk.wav", still="speakers")
    new_chat(s)
    s.ask("Which note is this string playing, and is it in tune?", "tuner/guitar_a2.wav", still="tuner")


def voice(s: Stage) -> None:
    s.card("Voice clips", "Record a question with the mic, or attach speech in another language")
    s.record_clip("sounds/capital.wav")
    s.send("voice")
    new_chat(s)
    s.ask("Translate this into English.", "audio/languages/spanish.wav", still="translate")


def web(s: Stage) -> None:
    s.card("The web", "Searches the web and reads the pages it finds")
    s.ask("Search the web: what is the tallest building in the world?", still="web")


def thinking(s: Stage) -> None:
    s.card("Thinking", "Models that can think are given time to, picked from the same menu as the model")
    s.pick("Medium")
    s.ask("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much is the ball?",
          still="thinking")
    s.pick("Off")


def suggestion(s: Stage) -> None:
    s.card("Suggestions", "Each reply suggests a question to ask next; Tab writes it into the message box")
    s.ask("Tell me one interesting fact about octopuses.")
    s.point("#text")
    s.wait(1.5)
    s.click("#text")
    s.page.keyboard.press("Tab")
    s.wait(1.5)
    s.send("suggestion")


def live_voice(s: Stage) -> None:
    s.card("Live", "Talk, and it answers out loud, hands free")
    s.page.evaluate("url => __showcase.mic(url)", "/showcase-media/sounds/joke.wav")
    s.click("#live")
    heard = {"spoke": False}

    def done(state, quiet):
        status = s.page.evaluate("document.getElementById('livestatus').textContent")
        heard["spoke"] = heard["spoke"] or status == "Speaking"
        return heard["spoke"] and status == "Listening" and not state["speaking"] and quiet > 1.5

    s.watch(done, 180)
    s.still("live")
    s.click("#endlive")
    s.wait(1.5)


def search(s: Stage) -> None:
    s.card("Search", "Finds things said earlier in the chat")
    s.ask("What's in this photo?", "dog.png")
    s.ask("What breed is it?")
    s.type("dog", "#search")
    s.page.wait_for_selector("#results button:not([disabled])", timeout=60000)
    s.wait(1.0)
    s.still("search")
    s.click("#results button:not([disabled])")
    s.wait(2.0)


def outro(s: Stage) -> None:
    s.card("lite-assistant", "github.com/bdbarnett/lite-assistant", 5.0)


SCENES = [intro, clock, seeing, dog_distance, editing, reading, plant, documents, sounds, voice, web,
          thinking, suggestion, live_voice, search, outro]


def available_mb() -> int:
    with open("/proc/meminfo") as f:
        return next(int(line.split()[1]) // 1024 for line in f if line.startswith("MemAvailable"))


def main(names) -> None:
    chosen = [scene for scene in SCENES if not names or scene.__name__ in names]
    with Stage("chat", settings=SETTINGS) as s:
        for scene in chosen:
            started = time.monotonic()
            try:
                if scene not in (intro, outro):
                    new_chat(s)
                scene(s)
            except Exception as e:
                s.note(f"error in {scene.__name__}: {e}")
            s.note(f"{scene.__name__} took {time.monotonic() - started:.0f} s, {available_mb()} MB free")
    assemble("chat")


if __name__ == "__main__":
    main(sys.argv[1:])
