"""
Tests for tool folders, and for the example tool file, examples/tools/helpers.py: its tools
that need no agent, with the web faked, and ask_for_help and helper_answers,
with fake agents that record how they were run and answer as Claude Code's
JSON output does.
"""
import base64
import json
import os
import re
import stat
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from lite_assistant.engines import ToolCall
from lite_assistant.function_tools import ToolError
from lite_assistant.tools import Toolbox, definition, load, run
from lite_assistant.users import User

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "tools", "helpers.py")
ENGINES = types.SimpleNamespace(embed={}, depth={}, detect={}, sound={}, ocr={}, segment={}, face={}, pose={}, recognize={}, edit={}, speaker={}, pitch={}, rerank={}, stt={}, default_stt=None, default_detect=None)
with mock.patch("shutil.which", return_value=None), mock.patch("os.access", return_value=False):
    # Loading it finds no helpers here, but gives the module the tests use
    load(EXAMPLE, ENGINES)
helpers_tool = sys.modules["lite_assistant_tool_file_helpers"]
ClaudeCode, Helpers = helpers_tool.ClaudeCode, helpers_tool.Helpers

# A fake agent: claude answers that it will be sunny, any other that it will rain
FAKE_AGENT = """#!{python}
import json, os, sys, time
name = os.path.basename(sys.argv[0])
prompt = sys.stdin.read()
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps({{"helper": name, "args": sys.argv[1:], "prompt": prompt, "cwd": os.getcwd()}}) + "\\n")
mode = os.environ.get("FAKE_MODE_" + name, "answer")
if mode == "slow":
    time.sleep(1)
if mode == "crash":
    sys.exit(name + ": not logged in")
answer = "It will be sunny." if name == "claude" else "It will rain."
print(json.dumps({{"result": answer if mode != "error" else "API error",
                  "is_error": mode == "error",
                  "permission_denials": [{{"tool_name": "Bash"}}] if mode == "denied" else []}}))
"""


class Gemini(ClaudeCode):
    """ A second helper, run like Claude, to ask by name. """
    name = "Gemini"
    heard = re.compile(r"\bgemini\b", re.IGNORECASE)
    program = "gemini"


class FakeSTT:
    def transcribe(self, audio, language=None):
        return audio.decode()


def clip(text):
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(text.encode()).decode(), "format": "wav"}}


def user(content):
    return {"role": "user", "content": content}


def assistant(content):
    return {"role": "assistant", "content": content}


class HelpersTestCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.claude = self.script(folder.name, "claude", FAKE_AGENT)
        self.gemini = self.script(folder.name, "gemini", FAKE_AGENT)
        self.log = os.path.join(folder.name, "log")
        self.env("FAKE_LOG", self.log)
        self.helpers = Helpers([ClaudeCode(self.claude), Gemini(self.gemini)], FakeSTT())
        # The helpers' runs finish before their folder goes
        self.addCleanup(lambda: [job.done.wait(10) for job in list(self.helpers.jobs)])

    @staticmethod
    def script(folder, name, text):
        path = os.path.join(folder, name)
        with open(path, "w") as f:
            f.write(text.format(python=sys.executable))
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        return path

    def env(self, name, value):
        patch = mock.patch.dict(os.environ, {name: value})
        patch.start()
        self.addCleanup(patch.stop)

    def setting(self, name, value):
        patch = mock.patch.object(helpers_tool, name, value)
        patch.start()
        self.addCleanup(patch.stop)

    def runs(self, helper=None):
        if not os.path.exists(self.log):
            return []
        with open(self.log) as f:
            return [run for run in map(json.loads, f) if helper in (None, run["helper"])]

    def ask(self, *conversation, request="What's the weather in Bernie, MO?"):
        return self.helpers.ask_for_help(request, conversation=list(conversation))


class TestAskingIsExplicit(HelpersTestCase):
    def test_not_without_being_asked(self):
        output = self.ask(user("What's the weather in Bernie, MO?"))
        self.assertEqual(output["reply_with"], "I can't do that myself, but I can ask Claude or Gemini. Want me to?")
        self.assertEqual(self.runs(), [])

    def test_named(self):
        for said in ("Ask Claude about the weather in Bernie.", "claude, what's the forecast?"):
            self.assertEqual(self.ask(user(said))["answer"], "It will be sunny.")
        self.assertEqual(self.ask(user("Ask Gemini what the forecast is."))["answer"], "It will rain.")
        self.assertEqual((len(self.runs("claude")), len(self.runs("gemini"))), (2, 1))

    def test_the_first_named(self):
        self.assertEqual(self.ask(user("Ask Gemini, or Claude if it can't."))["helper"], "Gemini")

    def test_yes_to_an_offer(self):
        offer = assistant(self.helpers.offer)
        self.assertEqual(self.ask(user("Weather in Bernie?"), offer, user("Yes please"))["helper"], "Claude")
        self.assertEqual(self.ask(offer, user("Ask gemini"))["helper"], "Gemini")
        self.assertEqual(self.ask(assistant("Gemini couldn't answer. Want me to ask Claude instead?"),
                                  user("Sure."))["helper"], "Claude")
        for declined in ("No thanks", "nope", "Don't bother."):
            with self.subTest(declined=declined):
                self.assertIn("not_asked", self.ask(offer, user(declined)))
        self.assertEqual(len(self.runs()), 3)

    def test_an_older_offer_doesnt_count(self):
        conversation = [assistant("Want me to ask Claude?"), user("No."), assistant("OK."), user("What's the weather?")]
        self.assertIn("not_asked", self.ask(*conversation))

    def test_heard_in_a_voice_clip(self):
        self.assertIn("answer", self.ask(user([clip("Ask Claude what the weather is.")])))
        self.assertIn("not_asked", self.ask(user([clip("What's the weather?")])))

    def test_parts(self):
        self.assertIn("answer", self.ask(user([{"type": "input_text", "text": "Ask Claude"}])))

    def test_only_installed_helpers_are_offered(self):
        helpers = Helpers([Gemini(self.gemini)])
        self.assertEqual(helpers.offer, "I can't do that myself, but I can ask Gemini. Want me to?")
        self.assertIn("not_asked", helpers.ask_for_help("Weather?", [user("Ask Claude about the weather.")]))


class TestRunningClaude(HelpersTestCase):
    ASKED = user("Ask Claude: weather in Bernie, MO?")

    def test_command(self):
        self.ask(self.ASKED)
        [ran] = self.runs()
        args = ran["args"]
        self.assertEqual(args[:3], ["-p", "--output-format", "json"])
        # Only the web, and none of the user's settings or MCP servers
        self.assertEqual(args[args.index("--tools") + 1], "WebSearch,WebFetch")
        self.assertEqual(args[args.index("--allowedTools") + 1], "WebSearch,WebFetch")
        self.assertEqual(args[args.index("--setting-sources") + 1], "project")
        self.assertIn("--strict-mcp-config", args)
        self.assertIn("--append-system-prompt", args)
        self.assertEqual(args[args.index("--model") + 1], "sonnet")
        # The prompt goes in on stdin, with the user's own words when Gemma's request differs
        self.assertEqual(ran["prompt"], "What's the weather in Bernie, MO?\n\n"
                                        "The user's own words were: \"Ask Claude: weather in Bernie, MO?\"")
        # It runs in an empty folder of its own, gone once it's done
        self.assertIn("lite-assistant-helper-", ran["cwd"])
        self.assertFalse(os.path.exists(ran["cwd"]))

    def test_settings(self):
        self.setting("CLAUDE_TOOLS", ["Read", "Bash"])
        self.setting("CLAUDE_ALLOWED", ["Read", "Bash(git log:*)"])
        self.setting("CLAUDE_SETTINGS", "user")
        self.setting("CLAUDE_MODEL", None)
        self.setting("CLAUDE_DIR", os.path.dirname(self.log))
        self.ask(self.ASKED)
        ran = self.runs()[0]
        args = ran["args"]
        self.assertEqual(args[args.index("--tools") + 1], "Read,Bash")
        self.assertEqual(args[args.index("--allowedTools") + 1], "Read,Bash(git log:*)")
        self.assertEqual(args[args.index("--setting-sources") + 1], "user")
        self.assertNotIn("--model", args)
        self.assertEqual(ran["cwd"], os.path.dirname(self.log))

    def test_images_go_along(self):
        jpeg = b"\xff\xd8fake"
        content = [{"type": "text", "text": "Claude, what's wrong with this wiring?"},
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}]
        self.ask(user(content), request="What's wrong with this wiring?")
        ran = self.runs()[0]
        path = ran["prompt"].split("The user sent these images: ")[1]
        self.assertTrue(path.endswith("image1.jpg"))
        self.assertEqual(ran["args"][ran["args"].index("--add-dir") + 1], os.path.dirname(path))
        # Claude can read them, there
        self.assertEqual(ran["args"][ran["args"].index("--tools") + 1], "WebSearch,WebFetch,Read")
        # The images are deleted once Claude is done
        self.assertFalse(os.path.exists(os.path.dirname(path)))

    def test_slow_answer_comes_later(self):
        self.env("FAKE_MODE_claude", "slow")
        with mock.patch.object(helpers_tool, "WAIT", 0.1):
            output = self.ask(self.ASKED)
        self.assertEqual(output["status"], "Claude is still working on it.")
        working = self.helpers.helper_answers()
        self.assertEqual((working["answers"], working["working_on"][0]["request"]),
                         ([], "What's the weather in Bernie, MO?"))
        deadline = time.time() + 10
        while not self.helpers.jobs[0].done.is_set() and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.helpers.helper_answers()["answers"],
                         [{"helper": "Claude", "request": "What's the weather in Bernie, MO?",
                           "answer": "It will be sunny."}])
        # Each answer is given once
        self.assertEqual(self.helpers.helper_answers()["answers"], [])

    def test_only_admins_ask_and_each_gets_their_own_answers(self):
        said = user("Ask Claude about the weather in Bernie.")
        refused = self.helpers.ask_for_help("Weather?", [said], User("alice"))
        self.assertEqual(refused["reply_with"], "Sorry, only the server's admins can ask Claude.")
        self.assertEqual(self.runs(), [])
        self.env("FAKE_MODE_claude", "slow")
        with mock.patch.object(helpers_tool, "WAIT", 0.1):
            self.helpers.ask_for_help("Weather?", [said], User("alex", admin=True))
        self.helpers.jobs[0].done.wait(10)
        self.assertEqual(self.helpers.helper_answers(User("alice"))["answers"], [])
        self.assertEqual(self.helpers.helper_answers(User("alex", admin=True))["answers"][0]["answer"],
                         "It will be sunny.")

    def test_failures_offer_the_other_helper(self):
        for mode, error in (("error", "API error"), ("crash", "not logged in")):
            with self.subTest(mode=mode):
                self.env("FAKE_MODE_claude", mode)
                output = self.ask(self.ASKED)
                self.assertIn(error, output["error"])
                self.assertEqual(output["reply_with"], output["error"] + " Want me to ask Gemini instead?")
        # With no other helper, there's nothing to offer
        self.assertNotIn("reply_with", Helpers([ClaudeCode(self.claude)]).ask_for_help("Weather?", [self.ASKED]))

    def test_denied_tools_are_reported(self):
        self.env("FAKE_MODE_claude", "denied")
        self.assertEqual(self.ask(self.ASKED)["not_allowed"], "Claude wasn't allowed to use Bash.")

    def test_long_answers_are_cut(self):
        self.helpers.jobs.append(job := helpers_tool.Job("Claude", "Tell me everything", answer="word " * 1000))
        job.done.set()
        self.assertEqual(len(self.helpers.helper_answers()["answers"][0]["answer"]), helpers_tool.MAX_ANSWER + 1)

    def test_busy(self):
        self.env("FAKE_MODE_claude", "slow")
        with mock.patch.object(helpers_tool, "WAIT", 0):
            for _ in range(helpers_tool.MAX_RUNNING):
                self.assertIn("status", self.ask(self.ASKED))
            self.assertIn("already working", self.ask(user("Ask gemini"))["error"])


class TestNoAgent(unittest.TestCase):
    def test_calculate(self):
        for expression, result in (("(17.5 * 4) / 3", 23.33333333), ("2^10", 1024), ("1,000 × 3", 3000),
                                   ("sqrt(16) + pi", 7.141592654), ("round(2.567, 2)", 2.57), ("15 % 4", 3)):
            with self.subTest(expression=expression):
                self.assertEqual(helpers_tool.calculate(expression)["result"], result)
        for expression, error in (("__import__('os')", "Can't work out"), ("1/0", "ZeroDivisionError"),
                                  ("9**9**9", "OverflowError"), ("x" * 201, "too long"), ("2 +", "SyntaxError")):
            with self.subTest(expression=expression):
                self.assertIn(error, helpers_tool.calculate(expression)["error"])

    def test_convert_units(self):
        for value, source, target, result in ((5, "miles", "km", 8.04672), (2, "cups", "ml", 473.176),
                                              (72, "°F", "°C", 22.2222), (1, "fl oz", "tbsp", 2),
                                              (100, "degrees Celsius", "Fahrenheit", 212), (60, "mph", "km/h", 96.5606)):
            with self.subTest(source=source, target=target):
                self.assertEqual(helpers_tool.convert_units(value, source, target)["result"], result)
        self.assertIn("Can't convert", helpers_tool.convert_units(1, "mile", "kg")["error"])
        with self.assertRaisesRegex(ValueError, "Don't know the unit"):
            helpers_tool.convert_units(1, "furlongs", "m")

    def web(self, *answers):
        patch = mock.patch.object(helpers_tool, "fetch", side_effect=list(answers))
        fetch = patch.start()
        self.addCleanup(patch.stop)
        return fetch

    PLACES = {"results": [
        {"name": "Bernie", "admin1": "Texas", "country": "United States", "country_code": "US", "latitude": 1, "longitude": 2},
        {"name": "Bernie", "admin1": "Missouri", "country": "United States", "country_code": "US", "latitude": 36.67,
         "longitude": -89.97}]}
    FORECAST = {"current": {"temperature_2m": 50.7, "apparent_temperature": 48.6, "weather_code": 0, "wind_speed_10m": 1.4},
                "daily": {"time": ["2026-10-25", "2026-10-26", "2026-10-27"], "weather_code": [3, 61, 95],
                          "temperature_2m_max": [76.5, 75.8, 74.8], "temperature_2m_min": [47.1, 49.5, 55.3],
                          "precipitation_probability_max": [1, 60, 80]}}

    def test_get_weather(self):
        fetch = self.web(self.PLACES, self.FORECAST)
        weather = helpers_tool.get_weather("Bernie, MO")
        self.assertEqual(fetch.call_args_list[0].kwargs["name"], "Bernie")
        # The state picks which Bernie, and the US gets Fahrenheit
        self.assertEqual((fetch.call_args.kwargs["latitude"], fetch.call_args.kwargs["temperature_unit"]),
                         (36.67, "fahrenheit"))
        self.assertEqual(weather["place"], "Bernie, Missouri, United States")
        self.assertEqual(weather["now"], {"temperature": 51, "feels_like": 49, "weather": "clear", "wind": 1})
        self.assertEqual([(d["day"], d["weather"], d["high"]) for d in weather["days"]],
                         [("today", "cloudy", 76), ("tomorrow", "light rain", 76), ("Tuesday", "thunderstorms", 75)])

    def test_get_weather_needs_a_place(self):
        self.assertIn("Ask the user where", helpers_tool.get_weather()["error"])
        self.web({"results": []})
        with self.assertRaisesRegex(ValueError, "Couldn't find a place called Nowhere"):
            helpers_tool.get_weather("Nowhere")

    def test_look_up(self):
        fetch = self.web({"query": {"search": [{"title": "Bernie, Missouri"}]}},
                         {"extract": "Bernie is a city in southern Stoddard County, Missouri."})
        self.assertEqual(helpers_tool.look_up("Bernie MO"),
                         {"title": "Bernie, Missouri", "summary": "Bernie is a city in southern Stoddard County, Missouri."})
        self.assertTrue(fetch.call_args.args[0].endswith("/page/summary/Bernie%2C_Missouri"))
        self.web({"query": {"search": []}})
        self.assertIn("nothing on", helpers_tool.look_up("xyzzy")["error"])


class TestBuiltinWiring(HelpersTestCase):
    def test_conversation_is_passed_but_never_offered(self):
        self.assertEqual(list(definition(self.helpers.ask_for_help)["function"]["parameters"]["properties"]),
                         ["request"])
        call = ToolCall("ask_for_help", json.dumps({"request": "Weather?", "conversation": "made up by the model"}))
        output = json.loads(run(self.helpers.ask_for_help, call, [user("Ask Claude: weather?")])[0])
        self.assertEqual(output["answer"], "It will be sunny.")

    def test_tools(self):
        self.env("PATH", os.path.dirname(self.claude))
        tools = helpers_tool.tools(ENGINES)
        self.assertEqual([f.__name__ for f in tools], ["get_weather", "look_up", "calculate", "convert_units",
                                                        "ask_for_help", "helper_answers"])
        self.assertEqual([h.command for h in tools[-1].__self__.helpers], [self.claude])
        # Without Claude Code, the rest
        with mock.patch.object(ClaudeCode, "find", return_value=None):
            self.assertEqual(len(helpers_tool.tools(ENGINES)), 4)


DICE = ("def tools(engines):\n"
        "    def roll_dice() -> int:\n"
        "        \"\"\" Rolls a die. \"\"\"\n"
        "        return 4\n"
        "    return [roll_dice]\n")


class TestToolFolders(unittest.TestCase):
    def folder(self, **files):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        for name, text in files.items():
            with open(os.path.join(folder.name, name), "w") as f:
                f.write(text)
        return folder.name

    def create(self, *folders, default=None):
        with mock.patch.dict(os.environ):
            os.environ.pop("LITE_ASSISTANT_TOOLS", None)
            return Toolbox.create(ENGINES, default, folders)

    def test_tools_are_added_and_on(self):
        toolbox = self.create(self.folder(**{"dice.py": DICE, "notes.txt": "Not a tool file."}))
        self.assertEqual(toolbox.default, ["get_time", "load_skill", "search_web", "read_web_page", "roll_dice"])
        self.assertEqual(toolbox.functions["roll_dice"](), 4)
        # LITE_ASSISTANT_TOOLS still picks which are on
        self.assertEqual(self.create(self.folder(**{"dice.py": DICE}), default="get_time").default, ["get_time"])
        # A folder that isn't there, such as the owner's before they make one, adds none
        self.assertNotIn("roll_dice", self.create("/no/such/folder").functions)

    def test_a_broken_file_is_skipped(self):
        toolbox = self.create(self.folder(**{"dice.py": DICE, "broken.py": "def tools(engines):\n    return [",
                                             "none.py": "TOOLS = []\n", "_helper.py": "raise SystemExit"}))
        self.assertIn("roll_dice", toolbox.functions)

    def test_two_tools_with_one_name_are_both_left_out(self):
        clock = DICE.replace("roll_dice", "get_time")
        with mock.patch("builtins.print") as printed:
            toolbox = self.create(self.folder(**{"dice.py": DICE, "clock.py": clock}), self.folder(**{"more.py": DICE}))
        self.assertNotIn("roll_dice", toolbox.functions)
        said = " ".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertRegex(said, r"dice\.py and .*more\.py both have one")
        # A built-in keeps its name
        self.assertEqual(toolbox.functions["get_time"].__module__, "lite_assistant.tools.clock")
        self.assertIn("a built-in tool has that name", said)

    def test_a_tool_can_ask_who_is_asking(self):
        def whose_notes(user=None) -> str:
            """ Says whose notes these are. """
            return user.folder("notes") if user else "no one's"
        self.assertNotIn("user", definition(whose_notes)["function"]["parameters"]["properties"])
        alice = User("alice")
        output, _ = run(whose_notes, ToolCall("whose_notes", '{"user": "mallory"}'), [], alice)
        # The model can't pass a user of its own
        self.assertEqual(output, alice.folder("notes"))

    def test_bad_files(self):
        with self.assertRaisesRegex(ToolError, "no tools"):
            load(self.folder(**{"t.py": "TOOLS = []\n"}) + "/t.py", ENGINES)
        with self.assertRaisesRegex(ToolError, "isn't a Python file"):
            load("/no/such/tools.py", ENGINES)


if __name__ == "__main__":
    unittest.main()
