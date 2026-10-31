"""
The stage the showcase videos are made on: headless Chromium driving the
chat page in dark mode at 1080p, with everything the page plays recorded
alongside, and the stretches where nothing happens cut short when ffmpeg
puts the video together.

Needs Playwright (pip install playwright; playwright install chromium),
ffmpeg, a running server, and the clips and photos in test-media/.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
MEDIA = HERE.parent / "test-media"
OUTPUT = HERE / "output"
# The page at 1280x720, drawn at 1.5x, so the video and stills are 1080p
VIEWPORT, SCALE = {"width": 1280, "height": 720}, 1.5
FPS = 30
# A stretch longer than this with nothing changing on screen and nothing heard is cut short,
# keeping this much of its start and its end
IDLE_CUT, KEEP_START, KEEP_END = 3.0, 0.8, 0.6
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# The browser records nothing while nothing plays, leaving gaps in the recording's times: this fills them with silence
FILL = "aresample=48000:async=1:first_pts=0"
# What changes on screen as a reply comes, and whether anything is being said
STATE = """() => {
    const m = document.getElementById("messages");
    // speaker.chain settles when every sentence asked for has been fetched
    const c = speaker.chain;
    if (window.__chain !== c) { window.__chain = c; window.__chainDone = false;
                                c.then(() => { if (window.__chain === c) window.__chainDone = true; }); }
    return { seen: [m.innerText.length, m.querySelectorAll("img").length, document.getElementById("modelnote").textContent,
                    document.getElementById("livestatus").textContent, document.querySelectorAll(".note").length].join("|"),
             speaking: speaker.player.busy() || (typeof live !== "undefined" && !!live?.player?.busy?.()),
             fetching: !window.__chainDone, busy: !!busy };
}"""


# When the chat page is ready to use
CHAT_READY = "typeof loadModels === 'function' && document.getElementById('pickmodel').textContent"


class Stage:
    def __init__(self, name: str, url: str = "http://localhost:5000/", settings: Optional[dict] = None,
                 ready: str = CHAT_READY):
        """ A page at url, set up with the chat page's settings, recorded once ready is true on it. """
        self.name, self.url, self.settings, self.ready = name, url, settings or {}, ready
        self.out = OUTPUT / name
        (self.out / "stills").mkdir(parents=True, exist_ok=True)
        self.audio_path = self.out / "audio.webm"
        self.idle: List[Tuple[float, float]] = []
        self.stills = 0
        self.errors: List[str] = []

    # --- Starting and finishing -------------------------------------------------

    def __enter__(self) -> "Stage":
        self.audio_path.unlink(missing_ok=True)
        self.playwright = sync_playwright().start()
        # With no sound server, Chromium plays to a stand-in device: the recording still hears it, and the
        # machine's speakers don't, and WSLg's sound server gave the page's audio errors after a minute
        # The scale is the browser's own, not Playwright's emulated one, which the screencast ignores
        self.browser = self.playwright.chromium.launch(args=["--autoplay-policy=no-user-gesture-required",
                                                             f"--force-device-scale-factor={SCALE}"],
                                                       env={**os.environ, "PULSE_SERVER": "/nonexistent"})
        self.context = self.browser.new_context(viewport=VIEWPORT, color_scheme="dark")
        self.context.add_init_script(path=str(HERE / "hooks.js"))
        self.context.expose_function("__saveAudio", self.save_audio)
        # The page fetches the microphone's clips from here, and the showcase's own pages from beside this file
        self.context.route("**/showcase-media/**", self.serve_media)
        self.context.route("**/showcase-pages/**", self.serve_page)
        self.page = self.context.new_page()
        self.t0, self.wall0 = time.monotonic(), time.time()
        self.page.on("pageerror", lambda e: self.note(f"page error: {e}"))
        self.page.on("console", lambda m: m.type == "error" and self.note(f"console: {m.text}"))
        self.page.goto(self.url)
        self.page.evaluate("""settings => {
            localStorage.setItem("la.settings", JSON.stringify({...JSON.parse(localStorage.getItem("la.settings") || "{}"),
                                                                ...settings, version: 2}));
            localStorage.removeItem("la.conversation");
        }""", self.settings)
        self.page.reload()
        self.page.wait_for_function(self.ready)
        self.start_screencast()
        self.page.evaluate("__showcase.record()")
        self.wait(1.0)
        self.mark = self.now()
        self.page.evaluate("__showcase.mark()")
        self.wait(1.5)
        return self

    def __exit__(self, *exc):
        try:
            self.page.evaluate("__showcase.stop()")
        except Exception as e:
            self.note(f"couldn't stop the sound recording: {e}")
        video = self.stop_screencast()
        self.page.close()
        self.context.close()
        self.browser.close()
        self.playwright.stop()
        timeline = {"idle": self.idle, "mark": self.mark, "errors": self.errors, "video": str(video)}
        (self.out / "timeline.json").write_text(json.dumps(timeline, indent=1))
        self.note(f"recorded {video}")

    # --- The picture ----------------------------------------------------------------

    # Playwright's own video is the viewport's size, not the pixels drawn, so the frames come straight from
    # Chromium's screencast instead: at the browser's scale, each one when the page changes
    def start_screencast(self) -> None:
        self.frames_dir = self.out / "raw" / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        for old in self.frames_dir.glob("*.jpg"):
            old.unlink()
        self.frames: List[Tuple[float, str]] = []
        self.cdp = self.context.new_cdp_session(self.page)
        self.cdp.on("Page.screencastFrame", self.save_frame)
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92,
                                               "maxWidth": round(VIEWPORT["width"] * SCALE),
                                               "maxHeight": round(VIEWPORT["height"] * SCALE)})

    def save_frame(self, frame: dict) -> None:
        import base64
        name = f"{len(self.frames):06d}.jpg"
        (self.frames_dir / name).write_bytes(base64.b64decode(frame["data"]))
        self.frames.append((frame["metadata"]["timestamp"], name))
        self.latest = name
        try:
            self.cdp.send("Page.screencastFrameAck", {"sessionId": frame["sessionId"]})
        except Exception:
            pass

    def stop_screencast(self) -> Path:
        """ Turns the frames, each shown until the next, into a steady 30 frames a second. """
        self.cdp.send("Page.stopScreencast")
        listing = self.out / "raw" / "frames.txt"
        lines = []
        for (t, name), (after, _) in zip(self.frames, self.frames[1:] + [(time.time(), "")]):
            lines += [f"file 'frames/{name}'", f"duration {max(after - t, 0.001):.4f}"]
        listing.write_text("\n".join(lines + [f"file 'frames/{self.frames[-1][1]}'"]) + "\n")
        video = self.out / "raw" / "video.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-vf", f"fps={FPS}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                        "-pix_fmt", "yuv420p", str(video)], check=True)
        for name in self.frames_dir.glob("*.jpg"):
            name.unlink()
        return video

    # --- Sound and media ------------------------------------------------------------

    def save_audio(self, chunk: str) -> None:
        import base64
        with open(self.audio_path, "ab") as f:
            f.write(base64.b64decode(chunk))

    def serve_media(self, route) -> None:
        relative = route.request.url.split("/showcase-media/", 1)[1]
        path = (MEDIA / relative).resolve()
        if MEDIA.resolve() not in path.parents or not path.is_file():
            route.fulfill(status=404)
        else:
            route.fulfill(path=str(path))

    def serve_page(self, route) -> None:
        name = route.request.url.split("/showcase-pages/", 1)[1].split("?")[0]
        path = (HERE / name).resolve()
        if path.parent != HERE or path.suffix not in (".html", ".js", ".css") or not path.is_file():
            route.fulfill(status=404)
        else:
            route.fulfill(path=str(path))

    def note(self, text: str) -> None:
        line = f"[{self.now():7.1f}s] {text}"
        print(line, flush=True)
        if "error" in text:
            self.errors.append(line)
        with open(self.out / "log.txt", "a") as f:
            f.write(line + "\n")

    def now(self) -> float:
        return time.monotonic() - self.t0

    def wait(self, seconds: float) -> None:
        # Playwright's own wait, so the page's calls back to the script still run
        self.page.wait_for_timeout(seconds * 1000)

    # --- What the viewer sees ---------------------------------------------------

    def card(self, title: str, subtitle: str = "", seconds: float = 3.5) -> None:
        self.note(f"card: {title}")
        self.page.evaluate("([t, s]) => __showcase.card(t, s)", [title, subtitle])
        self.wait(seconds)
        self.page.evaluate("__showcase.uncard()")
        self.wait(0.8)

    def point(self, target) -> None:
        locator = self.page.locator(target).first if isinstance(target, str) else target
        locator.scroll_into_view_if_needed()
        box = locator.bounding_box()
        self.page.evaluate("([x, y]) => __showcase.point(x, y)", [box["x"] + box["width"] / 2, box["y"] + box["height"] / 2])
        self.wait(0.75)

    def click(self, target) -> None:
        locator = self.page.locator(target).first if isinstance(target, str) else target
        self.point(locator)
        self.page.evaluate("__showcase.press()")
        locator.click()
        self.wait(0.4)

    def type(self, text: str, selector: str = "#text") -> None:
        self.click(selector)
        self.page.locator(selector).press_sequentially(text, delay=28)
        self.wait(0.5)

    def attach(self, *files: str) -> None:
        self.click("#attach")
        self.page.set_input_files("#file", [str(MEDIA / f) for f in files])
        self.page.wait_for_function(f"document.querySelectorAll('#attachments .chip').length >= {len(files)}",
                                    timeout=30000)
        self.wait(1.2)

    def still(self, name: str) -> Path:
        self.stills += 1
        # The screencast's latest frame is the page as it is now, at the full scale screenshots don't get
        self.wait(0.3)
        path = self.out / "stills" / f"{self.stills:02d}-{name}.png"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.frames_dir / self.latest), str(path)],
                       check=True)
        self.note(f"still {path.name}")
        return path

    # --- Chatting -----------------------------------------------------------------

    def ask(self, text: str = "", *files: str, still: str = "") -> str:
        """ Sends a message, with files, and waits for the reply and for it to be spoken; returns the reply. """
        if files:
            self.attach(*files)
        if text:
            self.type(text)
        return self.send(still)

    def send(self, still: str = "") -> str:
        self.click("#send")
        return self.wait_reply(still)

    def wait_reply(self, still: str = "", timeout: float = 300) -> str:
        self.page.wait_for_function("!!busy", timeout=15000)
        self.watch(lambda s, quiet: not s["busy"] and not s["speaking"] and not s["fetching"] and quiet > 1.5, timeout)
        reply = self.page.evaluate("turns.filter(t => t.role === 'assistant').pop()?.el.innerText || ''")
        self.note("reply: " + " ".join(reply.split())[:300])
        self.wait(1.0)
        if still:
            self.still(still)
        return reply

    def watch(self, done, timeout: float) -> None:
        """
        Polls the page until done(state, seconds quiet) is true, noting each
        stretch longer than IDLE_CUT with nothing changing or heard, to cut.
        """
        started, last, quiet_since = self.now(), None, self.now()
        while True:
            state = self.page.evaluate(STATE)
            now = self.now()
            if state["seen"] != last or state["speaking"]:
                if now - quiet_since > IDLE_CUT:
                    self.idle.append((quiet_since, now))
                last, quiet_since = state["seen"], now
            if done(state, now - quiet_since):
                if now - quiet_since > IDLE_CUT:
                    self.idle.append((quiet_since, now))
                return
            if now - started > timeout:
                self.note(f"error: gave up waiting after {timeout:.0f} s")
                return
            self.wait(0.2)

    def record_clip(self, clip: str) -> None:
        """ Records a voice clip with the mic button, the microphone playing clip. """
        self.page.evaluate("url => __showcase.mic(url)", f"/showcase-media/{clip}")
        self.click("#mic")
        self.page.wait_for_function("__showcase.micEnds && Date.now() > __showcase.micEnds + 600", timeout=120000)
        self.click("#mic")
        self.page.wait_for_function("document.querySelectorAll('#attachments .chip').length > 0", timeout=10000)
        self.wait(1.0)

    def pick(self, label: str) -> None:
        """ Picks an item from the model pill's menu. """
        self.click("#pick")
        self.wait(0.8)
        self.click(self.page.locator("#pickmenu button", has_text=label))
        self.wait(0.6)


def flash_at(video: str) -> float:
    """ When the white flash shows in the video: the first frame far brighter than the dark page. """
    log = subprocess.run(["ffmpeg", "-v", "error", "-t", "60", "-i", video, "-vf",
                          "signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-", "-f", "null", "-"],
                         capture_output=True, text=True).stdout
    for time_, brightness in re.findall(r"pts_time:([0-9.]+)\s+lavfi.signalstats.YAVG=([0-9.]+)", log):
        if float(brightness) > 120:
            return float(time_)
    raise RuntimeError(f"No flash found in {video}")


def beep_at(audio: str) -> float:
    """ When the beep sounds in the recording: the end of any silence it starts with. """
    log = subprocess.run(["ffmpeg", "-v", "info", "-i", audio, "-af", f"{FILL},atrim=end=60,silencedetect=n=-30dB:d=0.3",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    starts = re.search(r"silence_start:\s*([0-9.]+)", log)
    if starts and float(starts.group(1)) < 0.05:
        found = re.search(r"silence_end:\s*([0-9.]+)", log)
        if not found:
            raise RuntimeError(f"No beep found in {audio}")
        return float(found.group(1))
    return 0.0


def assemble(name: str) -> Path:
    """ The video: the recording with its sound, idle stretches cut short with a badge saying how long they were. """
    out = OUTPUT / name
    timeline = json.loads((out / "timeline.json").read_text())
    video, audio = timeline["video"], out / "audio.webm"
    has_audio = audio.exists() and audio.stat().st_size > 0
    # The white flash and the beep happened at the script's time "mark": where they fall says where
    # the script's times fall in the video, and how far to move the sound to meet it
    to_video = flash_at(video) - timeline["mark"]
    to_audio = beep_at(str(audio)) - timeline["mark"] if has_audio else 0
    start = timeline["mark"] + to_video + 1.0
    # Everything before the flash ended is left out, and each long wait is cut short
    cuts = [(0.0, start, 0.0)] + [(a + to_video + KEEP_START, b + to_video - KEEP_END, b - a)
                                  for a, b in timeline["idle"] if b - a >= IDLE_CUT and a + to_video > start]
    keep = "*".join(f"not(between(t,{s:.3f},{e:.3f}))" for s, e, _ in cuts)
    # One badge for each wait, and waits a moment apart, like before and after a tool's call, count as one
    waits, removed = [], 0.0
    for s, e, length in cuts:
        at = s - removed
        removed += e - s
        if not length:
            continue
        if waits and at - waits[-1][1] < 2.5:
            waits[-1][1], waits[-1][2] = at, waits[-1][2] + length + (at - waits[-1][1])
        else:
            waits.append([at, at, length])
    badges = [f"drawtext=fontfile={FONT}:text='waited {length:.0f} s':fontsize=30:fontcolor=white:"
              f"box=1:boxcolor=black@0.65:boxborderw=14:x=w-tw-48:y=40:enable='between(t,{first:.2f},{last + 2.2:.2f})'"
              for first, last, length in waits]
    length = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                                  capture_output=True, text=True, check=True).stdout.strip())
    delay = to_video - to_audio
    line_up = f"adelay={delay * 1000:.0f}:all=1" if delay >= 0 else f"atrim=start={-delay:.3f},asetpts=PTS-STARTPTS"
    # Kept frames and samples are numbered afresh, so what's left plays without gaps
    graph = (f"[0:v]fps={FPS},select='{keep}',setpts=N/{FPS}/TB" + "".join("," + b for b in badges) + "[v];"
             + (f"[1:a]{FILL},{line_up},apad=whole_dur={length:.3f},atrim=end={length:.3f},"
                f"aselect='{keep}',asetpts=N/SR/TB[a]" if has_audio else "anullsrc=r=48000:cl=stereo[a]"))
    (out / "graph.txt").write_text(graph)
    result = out / f"{name}.mp4"
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", timeline["video"]] + (["-i", str(audio)] if has_audio else []) + \
        ["-filter_complex_script", str(out / "graph.txt"), "-map", "[v]", "-map", "[a]", "-shortest",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(result)]
    subprocess.run(command, check=True)
    total = sum(e - s for s, e, _ in cuts[1:])
    print(f"{result}: {len(cuts) - 1} waits cut, {total:.0f} s shorter", flush=True)
    return result


if __name__ == "__main__":
    # python stage.py NAME: puts NAME's video together again from its recording
    assemble(sys.argv[1])
