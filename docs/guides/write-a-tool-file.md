# Write a tool file

lite-assistant's built-in tools are plain Python functions the server runs itself when the model calls them (see [Built-in tools](../reference/built-in-tools.md)). A **tool file** adds more: a Python file in a tool folder. The server reads every tool file in `~/.config/lite-assistant/tools` when it starts, and in any folder you name with `--tools-dir`, which can be given more than once:

```bash
mkdir -p ~/.config/lite-assistant/tools
cp examples/tools/helpers.py ~/.config/lite-assistant/tools/
lite-assistant serve
# Or, without copying it
lite-assistant serve --tools-dir examples/tools
```

Tools from tool files are on by default, alongside `get_time`; `LITE_ASSISTANT_TOOLS` and a request's `builtin_tools` still pick which are on. A file that fails to load is skipped, saying why; two tools with the same name are both left out, naming both files; and files starting with `_` are left alone, for code your tools share.

A tool file runs as your own code, with your permissions, for every user of the server, so only put files you wrote or have read in a tool folder. Requests can't load one.

## Writing a tool file

The file needs a `tools(engines)` function that returns a list of functions. Each function's name, signature and docstring become its tool definition. The docstring's first paragraph describes the tool, and its `Args:` section describes the parameters. A parameter without a default is required. A function returns anything JSON can encode, or a string.

```python
def tools(engines):
    def roll_dice(sides: int = 6) -> dict:
        """
        Rolls a die.

        Args:
            sides: How many sides the die has.
        """
        import random
        return {"rolled": random.randint(1, sides)}
    return [roll_dice]
```

`engines` holds the loaded engines: `engines.stt`, `engines.tts`, `engines.llm` and `engines.embed` are dicts by id, and `engines.default_stt` names the default speech-to-text engine. `tools` can return an empty list, for instance when something a tool needs is missing.

Three more things, the first two for what small models need help with:

- **`conversation`:** a function with a parameter named `conversation` gets the conversation so far, as chat messages. The model never sees that parameter. Use it to check what the user actually said, rather than trust the model's arguments.
- **`user`:** a function with a parameter named `user` gets the [user](manage-users-and-keys.md) asking: `user.name`, `user.admin`, and `user.folder(...)`, their own folder in the data directory, such as `user.folder("notes")`. The model never sees it either. Give it a default of `None`, which is what it gets when no one in particular is asking, as in tests. Use it to keep each user's things apart, or to let only admins do something.
- **`reply_with`:** when every built-in the model calls in a round returns a dict with a `reply_with` string, the server ends the reply with that text and doesn't ask the model again. Use it for something that must be said word for word, such as an offer or a refusal. Small models don't reliably relay a tool's text.
- **`images`:** a dict with an `images` list of base64 PNGs or JPEGs shows them to the user, as `image_generation_call` items in Responses that the chat page draws; the model gets the rest of the dict without them. `edit_photo` uses it.

## The example

[`examples/tools/helpers.py`](../../examples/tools/helpers.py) gives the model help with what small models do badly or can't know. Four tools need no agent:

- **`get_weather`:** the weather now and for the next three days, anywhere, from [Open-Meteo](https://open-meteo.com). "Bernie, MO" or "Paris, France" picks among places of the same name. Places in the US get Fahrenheit and miles per hour; elsewhere, Celsius and kilometers per hour. Set `HOME` to the place it uses when you don't say.
- **`look_up`:** what Wikipedia says about a topic, from its summary. `WIKIPEDIA` sets the language.
- **`calculate`:** arithmetic, worked out rather than guessed: `+ - * / ** %`, parentheses, `sqrt`, `round`, `log`, the trigonometric functions and `pi`. It understands `^`, `×`, `÷` and `1,000`. It evaluates only arithmetic, never Python.
- **`convert_units`:** lengths, weights, volumes (cups and spoons too), speeds and temperatures.

`get_weather` and `look_up` send what's asked to those services, which need no keys or accounts.

### Asking Claude for help

Two more tools hand what the rest can't do, such as the news, to [Claude Code](https://claude.com/claude-code) (`claude -p`), if it's installed:

- **`ask_for_help`** asks Claude. It runs only when you ask for it: name Claude ("ask Claude what's in the news today"), or say yes when the model offers ("I can't do that myself, but I can ask Claude. Want me to?"). The model may offer, but never asks on its own: the tool refuses unless your last message names Claude, or answers an offer that did. Voice clips are checked by their Moonshine transcripts, which may mishear the name; "clawed" and "clod" are accepted too.
- **`helper_answers`** gives answers that took too long for the reply ("what did Claude say?"). Claude gets 25 seconds; after that it carries on in the background, and the model says so. At most two requests run at once, and each gets 10 minutes. Answers are kept in memory, so a restart loses them.

Images you send with the request go along too.

Claude runs with your login and uses your usage, so **only admins can ask it**; anyone else is told so, and each user gets only the answers to what they asked. Even so, an admin's key could be lost, so the example keeps Claude to looking things up on the web:

- **Each request runs in an empty folder of its own,** deleted afterwards. Images you send are saved there for Claude to see.
- **Claude gets only web search and fetch** (`--tools WebSearch,WebFetch`), plus reading files in its own folder when you send images. Run headless, it can't ask permission for anything else.
- **None of your Claude Code settings, hooks or plugins are loaded** (`--setting-sources project`, in an empty folder), **and no MCP servers** (`--strict-mcp-config`).
- **It runs on Sonnet,** which costs about two cents a question at list price.

If the server only needs to serve this machine, `lite-assistant serve --host 127.0.0.1` keeps other machines from reaching it, and Claude. Otherwise, other machines need a [key](manage-users-and-keys.md).

### Changing it

To let Claude do more, such as read your repos, copy the example to `~/.config/lite-assistant/tools`, change the settings at its top, and restart the server:

| Setting | Example's value | |
|---|---|---|
| `HOME` | `None`, which asks | Where `get_weather` looks when you don't say, e.g. `"Bernie, MO"` |
| `WIKIPEDIA` | `"en"` | Wikipedia's language |
| `CLAUDE_TOOLS` | `["WebSearch", "WebFetch"]` | The tools Claude has at all (`--tools`) |
| `CLAUDE_ALLOWED` | `["WebSearch", "WebFetch"]` | Those it may use without asking (`--allowedTools`), e.g. `"Bash(git log:*)"` |
| `CLAUDE_DIR` | `None`, an empty folder of its own | The folder Claude starts in. It can read files there without asking |
| `CLAUDE_SETTINGS` | `"project"`, which loads none | Which of your Claude Code settings it loads: `user`, `project`, `local` |
| `CLAUDE_MODEL` | `"sonnet"` | `None` for your Claude Code default |
| `INSTRUCTIONS` | Answer briefly, for speech, and only look | What Claude is told with each request |

Leave out a tool by removing it from `tools()`.

### Adding another agent

A helper is a subclass of `Helper` with a `name`, a `heard` pattern for the name, a `program`, a `run` method that runs it on a prompt in a folder, and an `answer` method that reads its output. Add it to the `found` list in `tools()`. With more than one, Gemma's offer names them all, you pick one by name ("yes" picks the first), and when one can't answer, for instance when its usage limit is reached, the reply offers the next: "Antigravity couldn't answer: Individual quota reached. Resets in 16h. Want me to ask Claude instead?"

Check what an agent can do before you add it. Antigravity (`agy`), for instance, can't be kept from your own allow rules in `~/.gemini/antigravity-cli/settings.json`. Its `--sandbox` stops its commands from writing, but not from reading, and its file tool may still write in its own folder and in `/tmp`. A broad allow rule lets it read any file:

```python
class Antigravity(Helper):
    name = "Antigravity"
    heard = re.compile(r"\b(anti[- ]?gravity|agy)\b", re.IGNORECASE)
    program = "agy"

    def run(self, prompt, folder, images):
        # agy reads no prompt from stdin, and has no system prompt option
        command = [self.command, "--output-format", "json", "--print-timeout", f"{TIMEOUT}s", "--sandbox",
                   f"-p={INSTRUCTIONS}\n\n{prompt}"]
        return subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT + 30, cwd=folder)

    def answer(self, finished):
        output = self.json(finished)
        if output.get("status") != "SUCCESS":
            raise RuntimeError(output.get("error") or output.get("status") or "it gave no reason.")
        return (output.get("response") or "").strip(), [d.get("display_name") or d.get("action") or "a tool"
                                                       for d in output.get("denied_actions") or []]
```
