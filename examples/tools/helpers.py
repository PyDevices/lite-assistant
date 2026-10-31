"""
A tool file of helpers for Gemma, for what small models do badly or can't
know. Start the server with it to turn them on:

    lite-assistant serve --tools-dir examples/tools

or copy it to ~/.config/lite-assistant/tools, which the server always reads.

- get_weather: the weather now and the next few days, from Open-Meteo.
- look_up: a topic's summary, from Wikipedia.
- calculate: arithmetic, worked out rather than guessed.
- convert_units: lengths, weights, volumes, speeds and temperatures.
- ask_for_help and helper_answers: ask Claude Code (claude -p), if it's
  installed, for what the rest can't do, such as the news.

get_weather and look_up send what's asked to those services, which need no
keys or accounts. Claude runs only when the user names it ("ask Claude what's
in the news"), or says yes to Gemma's offer to ask: Gemma may offer, but never
asks on its own. Claude runs with your login and uses your usage, and anyone
who can reach the server can ask it, so this file keeps it to looking things
up on the web, in an empty folder of its own, with none of your Claude Code
settings, hooks or MCP servers. To let it do more, such as read your repos,
copy this file and change the settings below. See docs/guides/write-a-tool-file.md.
"""
import ast
import base64
import datetime
import json
import math
import operator
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from lite_assistant.media import MediaError, decode_data_url, image_url

# --- Settings -----------------------------------------------------------------

# Where get_weather looks when the user doesn't say, e.g. "Bernie, MO"; None to ask
HOME = None
# Wikipedia's language, for look_up
WIKIPEDIA = "en"
# Claude's tools: those it has at all, and those it may use without asking, since run headless, it can't ask
CLAUDE_TOOLS = ["WebSearch", "WebFetch"]
CLAUDE_ALLOWED = ["WebSearch", "WebFetch"]
# The folder Claude starts in; None for an empty one of its own
CLAUDE_DIR = None
# Which of your Claude Code settings it loads (user, project, local); in its own folder, "project" loads none
CLAUDE_SETTINGS = "project"
# Claude's model; None for your Claude Code default. Sonnet answers quicker, for less
CLAUDE_MODEL = "sonnet"
# What helpers are told with each request
INSTRUCTIONS = (
    "You're answering a request relayed by lite-assistant, a voice assistant whose small local model passed it on. "
    "Your answer will be read aloud: reply in a few short, plain sentences, with no markdown, lists, links or code. "
    "You can't ask the user anything, so make reasonable assumptions, and mention one only if the answer depends "
    "on it. You may only look things up and read; don't change anything.")
# Seconds a request waits for a quick answer before the helper carries on in the background
WAIT = 25
# Seconds a helper may take at most
TIMEOUT = 600
# Requests the helpers work on at once
MAX_RUNNING = 2
# Characters of an answer Gemma gets, since its context is small
MAX_ANSWER = 1500


def tools(engines) -> list:
    """ The tool file's tools: those that need no agent, and asking Claude if it's installed. """
    functions = [get_weather, look_up, calculate, convert_units]
    found = [helper for helper in (ClaudeCode.find(),) if helper]
    if not found:
        print("[Helpers] Claude Code isn't installed, so there's no one to ask for help.")
        return functions
    # Voice clips are transcribed to hear whether the user asked for a helper
    helpers = Helpers(found, engines.stt.get(engines.default_stt))
    return functions + [helpers.ask_for_help, helpers.helper_answers]


# --- Tools that need no agent -------------------------------------------------------

def fetch(url: str, **query) -> Any:
    """ JSON from a web API. """
    request = urllib.request.Request(url + "?" + urllib.parse.urlencode(query),
                                     headers={"User-Agent": "lite-assistant (github.com/bdbarnett/lite-assistant)"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)
    except OSError as exc:
        raise RuntimeError(f"Couldn't reach {urllib.parse.urlparse(url).hostname}: {getattr(exc, 'reason', exc)}")


US_STATES = dict(re.findall(r"(\w\w):([^,]+)", (
    "AL:Alabama,AK:Alaska,AZ:Arizona,AR:Arkansas,CA:California,CO:Colorado,CT:Connecticut,DE:Delaware,"
    "DC:District of Columbia,FL:Florida,GA:Georgia,HI:Hawaii,ID:Idaho,IL:Illinois,IN:Indiana,IA:Iowa,KS:Kansas,"
    "KY:Kentucky,LA:Louisiana,ME:Maine,MD:Maryland,MA:Massachusetts,MI:Michigan,MN:Minnesota,MS:Mississippi,"
    "MO:Missouri,MT:Montana,NE:Nebraska,NV:Nevada,NH:New Hampshire,NJ:New Jersey,NM:New Mexico,NY:New York,"
    "NC:North Carolina,ND:North Dakota,OH:Ohio,OK:Oklahoma,OR:Oregon,PA:Pennsylvania,RI:Rhode Island,"
    "SC:South Carolina,SD:South Dakota,TN:Tennessee,TX:Texas,UT:Utah,VT:Vermont,VA:Virginia,WA:Washington,"
    "WV:West Virginia,WI:Wisconsin,WY:Wyoming")))
# WMO weather codes, as Open-Meteo gives them
WEATHER = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "cloudy", 45: "foggy", 48: "foggy with frost",
           51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
           61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
           71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
           82: "heavy showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorms",
           96: "thunderstorms with hail", 99: "thunderstorms with hail"}


def find_place(place: str) -> Dict[str, Any]:
    """ Open-Meteo's best match for a place, such as "Bernie, MO" or "Paris, France". """
    name, _, within = (part.strip() for part in place.partition(","))
    found = fetch("https://geocoding-api.open-meteo.com/v1/search", name=name, count=10, language="en",
                  format="json").get("results") or []
    if within:
        wanted = {within.lower(), US_STATES.get(within.upper(), within).lower()}
        found = [f for f in found if wanted & {str(f.get(k, "")).lower() for k in ("admin1", "country", "country_code")}]
    if not found:
        raise ValueError(f"Couldn't find a place called {place}.")
    return found[0]


def get_weather(place: str = "") -> Dict[str, Any]:
    """
    The weather now and for the next three days, somewhere in the world.

    Args:
        place: A city or town, with its state or country if there are others of the same name, e.g. "Bernie, MO".
    """
    place = place.strip() or HOME
    if not place:
        return {"error": "No place given. Ask the user where."}
    found = find_place(place)
    us = found.get("country_code") == "US"
    forecast = fetch("https://api.open-meteo.com/v1/forecast", latitude=found["latitude"],
                     longitude=found["longitude"], timezone="auto", forecast_days=3,
                     current="temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                     daily="weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                     temperature_unit="fahrenheit" if us else "celsius", wind_speed_unit="mph" if us else "kmh")
    now, daily = forecast["current"], forecast["daily"]
    days = []
    for n, date in enumerate(daily["time"]):
        days.append({"day": "today" if n == 0 else "tomorrow" if n == 1
                     else datetime.date.fromisoformat(date).strftime("%A"),
                     "weather": WEATHER.get(daily["weather_code"][n], "unknown"),
                     "high": round(daily["temperature_2m_max"][n]), "low": round(daily["temperature_2m_min"][n]),
                     "chance_of_rain": f"{daily['precipitation_probability_max'][n]} percent"})
    return {"place": ", ".join(str(found[k]) for k in ("name", "admin1", "country") if found.get(k)),
            "units": "degrees Fahrenheit, miles per hour" if us else "degrees Celsius, kilometers per hour",
            "now": {"temperature": round(now["temperature_2m"]), "feels_like": round(now["apparent_temperature"]),
                    "weather": WEATHER.get(now["weather_code"], "unknown"), "wind": round(now["wind_speed_10m"])},
            "days": days}


def look_up(topic: str) -> Dict[str, Any]:
    """
    What Wikipedia says about a topic: a person, place, thing or event.

    Args:
        topic: What to look up, e.g. "Bernie, Missouri" or "photosynthesis".
    """
    api = f"https://{WIKIPEDIA}.wikipedia.org"
    found = fetch(f"{api}/w/api.php", action="query", list="search", srsearch=topic, srlimit=1,
                  format="json")["query"]["search"]
    if not found:
        return {"error": f"Wikipedia has nothing on {topic}."}
    title = found[0]["title"]
    page = fetch(f"{api}/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_"), safe=""))
    summary = page.get("extract") or ""
    return {"title": title, "summary": summary if len(summary) <= MAX_ANSWER else summary[:MAX_ANSWER] + "…"}


OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
             ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos,
             # Powers are worked out in floats, so a huge one overflows rather than running for ever
             ast.Pow: lambda a, b: float(a) ** b}
FUNCTIONS = {"sqrt": math.sqrt, "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
             "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos,
             "atan": math.atan, "degrees": math.degrees, "radians": math.radians,
             "log": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp}
CONSTANTS = {"pi": math.pi, "e": math.e}


def calculate(expression: str) -> Dict[str, Any]:
    """
    Works out arithmetic exactly: + - * / ** % and parentheses, sqrt, abs, round, floor, ceil, log, log10, exp,
    sin, cos and tan (in radians), and pi. Use it for any sum rather than working it out yourself.

    Args:
        expression: The arithmetic, e.g. "(17.5 * 4) / 3" or "sqrt(2) * 10".
    """
    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](value(node.left), value(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](value(node.operand))
        if isinstance(node, ast.Name) and node.id in CONSTANTS:
            return CONSTANTS[node.id]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS
                and not node.keywords):
            return FUNCTIONS[node.func.id](*map(value, node.args))
        raise ValueError(f"Can't work out {ast.unparse(node)}.")

    # Spoken and typed arithmetic, as Python writes it: 2^3, 3×4, 8÷2, 1,000
    written = re.sub(r"(?<=\d),(?=\d{3})", "", expression.replace("^", "**").replace("×", "*").replace("÷", "/"))
    if len(written) > 200:
        return {"error": "That's too long to work out."}
    try:
        result = value(ast.parse(written, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        return {"error": str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: {exc}"}
    if isinstance(result, float):
        result = int(result) if result.is_integer() and abs(result) < 1e15 else float(f"{result:.10g}")
    return {"expression": expression, "result": result}


# Each unit's kind and size in that kind's base unit: meters, kilograms, liters and meters per second
UNITS = {}
for kind, size, names in [
        ("length", 1, "m meter meters metre metres"), ("length", 1000, "km kilometer kilometers kilometre kilometres"),
        ("length", 0.01, "cm centimeter centimeters"), ("length", 0.001, "mm millimeter millimeters"),
        ("length", 1609.344, "mi mile miles"), ("length", 0.9144, "yd yard yards"),
        ("length", 0.3048, "ft foot feet"), ("length", 0.0254, "in inch inches"),
        ("mass", 1, "kg kilogram kilograms kilo kilos"), ("mass", 0.001, "g gram grams"),
        ("mass", 0.45359237, "lb lbs pound pounds"), ("mass", 0.028349523125, "oz ounce ounces"),
        ("mass", 6.35029318, "st stone stones"),
        ("volume", 1, "l liter liters litre litres"), ("volume", 0.001, "ml milliliter milliliters"),
        ("volume", 3.785411784, "gal gallon gallons"), ("volume", 0.946352946, "qt quart quarts"),
        ("volume", 0.473176473, "pt pint pints"), ("volume", 0.2365882365, "cup cups"),
        ("volume", 0.0295735295625, "floz fluidounce fluidounces"),
        ("volume", 0.01478676478125, "tbsp tablespoon tablespoons"),
        ("volume", 0.00492892159375, "tsp teaspoon teaspoons"),
        ("speed", 1, "m/s mps meterpersecond meterspersecond"),
        ("speed", 1 / 3.6, "km/h kph kmh kilometerperhour kilometersperhour"),
        ("speed", 0.44704, "mph mileperhour milesperhour"), ("speed", 0.514444, "kn knot knots")]:
    UNITS.update(dict.fromkeys(names.split(), (kind, size)))
# Each temperature scale, to and from Celsius
TEMPERATURES = {"c": (lambda c: c, lambda c: c), "f": (lambda f: (f - 32) / 1.8, lambda c: c * 1.8 + 32),
                "k": (lambda k: k - 273.15, lambda c: c + 273.15)}


def unit(name: str) -> str:
    """ A unit's name as UNITS and TEMPERATURES list it: "Fluid Ounces" is "fluidounces", "°F" is "f". """
    name = re.sub(r"[\s.°_-]|^degrees?", "", name.lower()).replace("fl", "fluid", 1).replace("fluiduid", "fluid")
    name = {"fluidoz": "floz", "celsius": "c", "centigrade": "c", "fahrenheit": "f", "kelvin": "k"}.get(name, name)
    if name not in UNITS and name not in TEMPERATURES:
        raise ValueError(f"Don't know the unit {name!r}.")
    return name


def convert_units(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
    """
    Converts between units of length, weight, volume (including cups and spoons), speed or temperature.

    Args:
        value: How many of from_unit.
        from_unit: The unit it's in, e.g. "miles", "cups", "°F".
        to_unit: The unit to convert it to, e.g. "km", "ml", "°C".
    """
    source, target = unit(from_unit), unit(to_unit)
    if source in TEMPERATURES and target in TEMPERATURES:
        result = TEMPERATURES[target][1](TEMPERATURES[source][0](value))
    elif source in UNITS and target in UNITS and UNITS[source][0] == UNITS[target][0]:
        result = value * UNITS[source][1] / UNITS[target][1]
    else:
        return {"error": f"Can't convert {from_unit} to {to_unit}."}
    return {"value": value, "from": from_unit, "to": to_unit, "result": float(f"{result:.6g}")}


# --- Asking Claude for help ---------------------------------------------------------

DECLINED = re.compile(r"^\W*(no|nope|nah|don'?t|do not|never|not now|stop|cancel)\b", re.IGNORECASE)


def find(name: str) -> Optional[str]:
    """ A helper's command, on PATH or in ~/.local/bin, which a systemd service's PATH leaves out. """
    local = os.path.expanduser(f"~/.local/bin/{name}")
    return shutil.which(name) or (local if os.access(local, os.X_OK) else None)


class Helper:
    """
    An agent run headless: how it's named, how it's run, and how it answers.
    Add another by subclassing this, and finding it in tools().
    """
    name = ""
    # The name, as speech recognition may hear it
    heard = re.compile("")
    program = ""

    def __init__(self, command: str):
        self.command = command

    @classmethod
    def find(cls) -> Optional["Helper"]:
        command = find(cls.program)
        return cls(command) if command else None

    def run(self, prompt: str, folder: str, images: List[str]) -> subprocess.CompletedProcess:
        """ Runs the helper on a prompt, in a folder of its own that holds the images. """
        raise NotImplementedError

    def answer(self, finished: subprocess.CompletedProcess) -> Tuple[str, List[str]]:
        """ The answer in a finished run's output, and the tools it was denied; RuntimeError if it failed. """
        raise NotImplementedError

    @staticmethod
    def json(finished: subprocess.CompletedProcess) -> Dict[str, Any]:
        try:
            return json.loads(finished.stdout)
        except ValueError:
            raise RuntimeError((finished.stderr or finished.stdout or f"exit code {finished.returncode}")
                               .strip()[-300:])


class ClaudeCode(Helper):
    name = "Claude"
    heard = re.compile(r"\b(claude|claud|clawed|clod)\b", re.IGNORECASE)
    program = "claude"

    def run(self, prompt, folder, images):
        # Read is for the images, in its folder; reading elsewhere needs permission it can't ask for
        tools = CLAUDE_TOOLS + (["Read"] if images and "Read" not in CLAUDE_TOOLS else [])
        command = [self.command, "-p", "--output-format", "json", "--permission-mode", "default",
                   "--no-session-persistence", "--strict-mcp-config", "--setting-sources", CLAUDE_SETTINGS,
                   "--append-system-prompt", INSTRUCTIONS, "--tools", ",".join(tools),
                   "--allowedTools", ",".join(CLAUDE_ALLOWED), "--add-dir", folder]
        if CLAUDE_MODEL:
            command += ["--model", CLAUDE_MODEL]
        # The prompt goes in on stdin, where options that take several values can't swallow it
        return subprocess.run(command, input=prompt, capture_output=True, text=True, timeout=TIMEOUT,
                              cwd=os.path.expanduser(CLAUDE_DIR) if CLAUDE_DIR else folder)

    def answer(self, finished):
        output = self.json(finished)
        if output.get("is_error"):
            raise RuntimeError(output.get("result") or "it gave no reason.")
        return (output.get("result") or "").strip(), [d.get("tool_name", "a tool")
                                                     for d in output.get("permission_denials") or []]


@dataclass
class Job:
    """ A request to a helper: done once it has an answer or an error. """
    helper: str
    request: str
    # The user who asked, who alone gets the answer
    asker: str = ""
    started: float = field(default_factory=time.time)
    answer: Optional[str] = None
    error: Optional[str] = None
    denied: List[str] = field(default_factory=list)
    reported: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def result(self) -> Dict[str, Any]:
        self.reported = True
        if self.error:
            return {"helper": self.helper, "request": self.request, "error": self.error}
        answer = self.answer if len(self.answer) <= MAX_ANSWER else self.answer[:MAX_ANSWER] + "…"
        found = {"helper": self.helper, "request": self.request, "answer": answer}
        if self.denied:
            found["not_allowed"] = f"{self.helper} wasn't allowed to use {', '.join(sorted(set(self.denied)))}."
        return found


class Helpers:
    """ Requests to the helpers, kept in memory, so a restart forgets them. """
    def __init__(self, helpers: List[Helper], stt=None):
        self.helpers, self.stt = helpers, stt
        self.jobs: List[Job] = []
        self.lock = threading.Lock()

    @property
    def offer(self) -> str:
        """ What Gemma says, in place of asking, when the user didn't ask for a helper. """
        return f"I can't do that myself, but I can ask {' or '.join(h.name for h in self.helpers)}. Want me to?"

    def ask_for_help(self, request: str, conversation: Optional[List[Dict[str, Any]]] = None,
                     user=None) -> Dict[str, Any]:
        """
        Asks Claude, a far more capable AI assistant that can search the web. Call it only when the user asks for
        Claude by name, or says yes after you offer to ask. When something is beyond you and your other tools,
        such as the news or anything current, offer to ask instead of calling this.

        Args:
            request: What to ask, in full, with anything it needs from the conversation.
        """
        said, offered, images = self.last_exchange(conversation or [])
        helper = self.named(said) or (self.offered(offered) if not DECLINED.match(said) else None)
        if not helper:
            # Gemma calls it anyway when it can't answer, which is the moment to offer
            return {"not_asked": "The user didn't ask for a helper, so none was asked.", "reply_with": self.offer}
        # A helper runs with the server owner's login, so only admins may ask one
        if user and not user.admin:
            return {"not_asked": "Only the server's admins can ask a helper.",
                    "reply_with": f"Sorry, only the server's admins can ask {helper.name}."}
        with self.lock:
            if sum(not job.done.is_set() for job in self.jobs) >= MAX_RUNNING:
                return {"error": f"The helpers are already working on {MAX_RUNNING} requests; "
                                 "try again when one is done."}
            job = Job(helper.name, request, user.name if user else "")
            self.jobs.append(job)
        threading.Thread(target=self.run, args=(helper, job, said, images), daemon=True).start()
        if not job.done.wait(WAIT):
            return {"status": f"{helper.name} is still working on it.",
                    "note": f"Tell the user {helper.name} is on it, and they can ask for its answer in a minute or two."}
        result = job.result()
        others = [h.name for h in self.helpers if h is not helper]
        if "error" in result and others:
            # A helper that couldn't do it is the moment to offer another
            result["reply_with"] = f"{result['error']} Want me to ask {others[0]} instead?"
        return result

    def helper_answers(self, user=None) -> Dict[str, Any]:
        """ Claude's answers to what you asked it earlier, and what it's still working on. """
        asker = user.name if user else ""
        with self.lock:
            mine = [job for job in self.jobs if job.asker == asker]
            finished = [job.result() for job in mine if job.done.is_set() and not job.reported]
            working = [{"helper": job.helper, "request": job.request,
                        "for": f"{round(time.time() - job.started)} seconds"}
                       for job in mine if not job.done.is_set()]
            # A finished request is reported once, then forgotten
            self.jobs = [job for job in self.jobs if not job.reported]
        if not finished and not working:
            return {"answers": [], "note": "There are no answers waiting, and nothing is being worked on."}
        return {"answers": finished, "working_on": working}

    def named(self, text: str) -> Optional[Helper]:
        """ The helper text names first, if any. """
        found = [(match.start(), n) for n, h in enumerate(self.helpers) if (match := h.heard.search(text))]
        return self.helpers[min(found)[1]] if found else None

    def offered(self, text: str) -> Optional[Helper]:
        """
        The helper an offer would ask: the one after "ask", as in "Want me to
        ask Claude instead?", or else the first it names.
        """
        for match in re.finditer(r"\bask\s+(.*)", text, re.IGNORECASE):
            helper = self.named(match.group(1)[:20])
            if helper:
                return helper
        return self.named(text)

    def last_exchange(self, conversation: List[Dict[str, Any]]) -> Tuple[str, str, List[Tuple[str, bytes]]]:
        """
        What the user said last, what the assistant said before it, and the
        images the user sent with it (media type and bytes). Voice clips count
        by their transcripts.
        """
        users = [i for i, message in enumerate(conversation) if message.get("role") == "user"]
        if not users:
            return "", "", []
        last = conversation[users[-1]]
        before = next((m for m in reversed(conversation[:users[-1]]) if m.get("role") == "assistant"), {})
        images = []
        for part in last.get("content") if isinstance(last.get("content"), list) else []:
            if part.get("type") in ("image_url", "input_image"):
                try:
                    url = image_url(part) if part.get("type") == "image_url" else part.get("image_url")
                    if isinstance(url, str) and url.startswith("data:"):
                        images.append(decode_data_url(url))
                except MediaError:
                    pass
        return self.text(last), self.text(before), images

    def text(self, message: Dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content
        said = []
        for part in content or []:
            if part.get("type") in ("text", "input_text", "output_text"):
                said.append(part.get("text") or "")
            elif part.get("type") == "input_audio" and self.stt:
                said.append(self.stt.transcribe(base64.b64decode(part["input_audio"]["data"])))
        return " ".join(said)

    def run(self, helper: Helper, job: Job, said: str, images: List[Tuple[str, bytes]]):
        folder = tempfile.mkdtemp(prefix="lite-assistant-helper-")
        prompt = job.request
        if said.strip() and said.strip() != job.request.strip():
            prompt += f'\n\nThe user\'s own words were: "{said.strip()}"'
        names = []
        for number, (media_type, data) in enumerate(images, 1):
            name = os.path.join(folder, f"image{number}.{media_type.split('/')[-1].replace('jpeg', 'jpg')}")
            with open(name, "wb") as f:
                f.write(data)
            names.append(name)
        if names:
            prompt += "\n\nThe user sent these images: " + ", ".join(names)
        print(f"[{helper.name}] Asked: {job.request}")
        try:
            job.answer, job.denied = helper.answer(helper.run(prompt, folder, names))
            if not job.answer:
                # A helper that was refused every tool it tried may have nothing to say
                raise RuntimeError(f"it wasn't allowed to use {', '.join(sorted(set(job.denied)))}."
                                   if job.denied else "it gave no answer.")
        except subprocess.TimeoutExpired:
            job.error = f"{helper.name} took longer than {TIMEOUT // 60} minutes, so it was stopped."
        except Exception as exc:
            reason = str(exc).strip()
            job.answer, job.error = None, f"{helper.name} couldn't answer: {reason}" + (
                "" if reason.endswith((".", "!", "?")) else ".")
        finally:
            shutil.rmtree(folder, ignore_errors=True)
            print(f"[{helper.name}] {'Answered' if job.answer else 'Failed'} after {time.time() - job.started:.0f}s: "
                  f"{(job.answer or job.error)[:200]}")
            job.done.set()
