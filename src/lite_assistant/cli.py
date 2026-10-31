"""
The lite-assistant command: 'serve' runs the server, 'setup' downloads the
models, 'models' lists, finds, adds and removes LiteRT-LM chat models, and
'users' adds and removes the users who may use the server, with their keys.
"""
import argparse
import os
import sys


def disable_power_throttling():
    """
    Windows throttles processes it judges to be in the background (EcoQoS),
    which includes a server with no visible window. It made the Gemma engine
    about 3x slower to read prompts (docs/how-it-works/benchmarks.md), so opt out.
    """
    import ctypes
    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]
    # Take control of EXECUTION_SPEED and IGNORE_TIMER_RESOLUTION, and turn both off
    state = PowerThrottlingState(1, 0x1 | 0x4, 0)
    current_process = ctypes.c_void_p(-1)
    process_power_throttling = 4
    if not ctypes.windll.kernel32.SetProcessInformation(
            current_process, process_power_throttling, ctypes.byref(state), ctypes.sizeof(state)):
        print("Warning: could not opt out of Windows power throttling.")


def serve(args):
    import uvicorn
    from . import paths
    from .app import create_app
    if sys.platform == "win32":
        disable_power_throttling()
    print(f"lite-assistant listening on http://{args.host}:{args.port}")
    if args.behind_proxy:
        print("Behind a proxy: every request needs a key, even from this PC.")
    if args.allow_public:
        print("Public addresses allowed: anyone on the internet who can reach this port can try keys.")
    # Notes from before there were users are the owner's
    from .users import Users
    old, notes = paths.data_dir("notes"), Users().owner().folder("notes")
    if os.path.isdir(old) and not os.path.exists(notes):
        os.makedirs(os.path.dirname(notes), exist_ok=True)
        os.rename(old, notes)
        print(f"Moved the notes in {old} to the owner's, {notes}")
    # The owner's tools folder is always read, if it's there
    tool_dirs = [paths.config_dir("tools")] + args.tools_dir
    for folder in args.tools_dir:
        if not os.path.isdir(folder):
            sys.exit(f"--tools-dir {folder} isn't a folder.")
    print(f"Tools from {', '.join(tool_dirs)}")
    if bool(args.certfile) != bool(args.keyfile):
        sys.exit("--certfile and --keyfile go together: the certificate and its private key.")
    for path in filter(None, (args.certfile, args.keyfile)):
        if not os.path.isfile(path):
            sys.exit(f"{path} isn't a file.")
    llm = args.llm or kept_llm()
    if llm == "none":
        print("No chat model: voice and the rest work, and chat answers with an error.")
    elif llm in ("e2b", "e4b"):
        from .engines import litertlm_engine
        litertlm_engine.MODEL = llm
    if args.forget_chats_after:
        print(f"Chats unchanged for {args.forget_chats_after:g} days are forgotten, but pinned ones.")
    uvicorn.run(create_app(tool_dirs=tool_dirs, behind_proxy=args.behind_proxy, allow_public=args.allow_public,
                           database=paths.data_dir("conversations.db"),
                           forget_chats_after=args.forget_chats_after, no_llm=llm == "none"),
                host=args.host, port=args.port, ssl_certfile=args.certfile or None, ssl_keyfile=args.keyfile or None)


LLM_CHOICES = ("e2b", "e4b", "none")


def kept_llm():
    """ The chat model setup kept, or None if it kept none. """
    import json
    from . import paths
    try:
        with open(paths.config_dir("setup.json"), encoding="utf-8") as f:
            return json.load(f).get("llm")
    except (FileNotFoundError, ValueError):
        return None


def llm_choice(args) -> str:
    """
    Which chat model setup downloads: "e2b", "e4b", "none" for boards only,
    or "auto" for the backend's own, E2B, and E4B too if a GPU works. --llm
    says, or else the answer given before; or else setup asks, in a terminal.
    With no terminal and no answer, as from a script, it's "auto". An answer
    is kept, so run.sh, which runs setup every time, asks only once.
    """
    import json
    from . import paths
    kept = paths.config_dir("setup.json")
    before = kept_llm()
    choice = args.llm or before
    if choice is None and sys.stdin.isatty():
        choice = ask_llm()
    if choice is None:
        return "auto"
    if choice != before:
        os.makedirs(os.path.dirname(kept), exist_ok=True)
        with open(kept, "w", encoding="utf-8") as f:
            json.dump({"llm": choice}, f)
    elif not args.llm:
        print(f"Chat model: {choice}, as chosen before; 'lite-assistant setup --llm e2b|e4b|none' changes it.")
    return choice


# E4B is recommended with this much memory: a server on it holds about 5 GB, E2B's 1.8 GB
E4B_MEMORY = 12e9


def memory() -> float:
    """ This machine's memory in bytes, or 0 if it can't be told. """
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("rest", ctypes.c_ulonglong * 6)]
            status = Status()
            status.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return float(status.total)
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return 0


def ask_llm() -> str:
    from .engines.litertlm_engine import MODELS
    roomy = memory() >= E4B_MEMORY
    marks = {"e2b": "", "e4b": ""}
    marks["e4b" if roomy else "e2b"] = " (recommended)"
    default = "2" if roomy else "1"
    print("Which chat model should setup download? The chat page needs one; boards don't, "
          "unless you want them to chat too.\n"
          f"  1) Gemma 4 E2B, {gigabytes(MODELS['e2b']['size'])}{marks['e2b']}: quick, even on a CPU, "
          "and a server on it holds 1.8 GB\n"
          f"  2) Gemma 4 E4B, {gigabytes(MODELS['e4b']['size'])}{marks['e4b']}: smarter, but half as fast "
          "without a GPU, and a server on it holds 5 GB\n"
          "  3) None, for boards only\n"
          "You can add more later with 'lite-assistant models', or 'Add or remove models…' in the chat page's "
          "model menu, or change this answer with 'lite-assistant setup --llm e2b|e4b|none'.")
    while True:
        try:
            answer = input(f"Choose 1, 2 or 3 [{default}]: ").strip() or default
        except EOFError:
            answer = default
        if answer in ("1", "2", "3"):
            return LLM_CHOICES[int(answer) - 1]


def setup(args):
    from .engines import engine_classes
    llm = llm_choice(args)
    print("Downloading models for every engine...")
    failed = []
    for registry in engine_classes().values():
        for engine_id, cls in registry.items():
            print(f"\n--- Setting up {engine_id} ---")
            try:
                if engine_id == "litertlm" and llm == "none":
                    print("No chat model, as chosen. To add one later: 'lite-assistant setup --llm e2b', "
                          "or 'lite-assistant models add', and restart the server.")
                    continue
                if engine_id == "litertlm" and llm != "auto":
                    cls.setup_model(llm)
                else:
                    cls.setup()
                print(f"Setup complete for {engine_id}.")
            except Exception as e:
                print(f"Failed to set up {engine_id}: {e}")
                failed.append(engine_id)
    from .vad import SileroVAD
    print("\n--- Setting up Silero VAD (Realtime turn detection) ---")
    try:
        SileroVAD.setup()
        print("Setup complete for Silero VAD.")
    except Exception as e:
        print(f"Failed to set up Silero VAD: {e}")
        failed.append("silero-vad")
    return 1 if failed else 0


def gigabytes(size) -> str:
    return "?" if size is None else f"{size / 1e9:.1f} GB"


def table(rows, headings):
    """ Prints rows of strings in columns under headings. """
    widths = [max(len(str(cell)) for cell in column) for column in zip(headings, *rows)]
    for row in [headings, *rows]:
        print("  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)).rstrip())


def models_list(args):
    from . import litert_models
    found = litert_models.downloaded()
    if not found:
        print("No LiteRT-LM models are downloaded. Find some with 'lite-assistant models search'.")
        return
    table([(model_id, gigabytes(os.path.getsize(path)), ", ".join(litert_models.facts(path)["modalities"]))
           for model_id, path in found.items()], ("MODEL", "SIZE", "TAKES"))


def models_search(args):
    from . import litert_models
    try:
        found = litert_models.search(" ".join(args.text) or None, limit=args.limit, sort=args.sort)
    except ValueError as e:
        print(e)
        return 1
    if not found:
        print("No LiteRT-LM chat models on Hugging Face match.")
        return
    fit = {True: "yes", False: "too big", None: "?"}
    table([(model["repo"], model["file"], gigabytes(model["size"]), fit[model["fits"]],
            "yes" if model["downloaded"] else "", model["downloads"]) for model in found],
          ("REPO", "FILE", "SIZE", "FITS", "HAVE", "DOWNLOADS"))
    print("\nAdd one with 'lite-assistant models add REPO [FILE]'.")


def models_add(args):
    from . import litert_models
    try:
        if args.repo.endswith(litert_models.SUFFIX) and os.path.isfile(args.repo):
            model_id = litert_models.add_file(args.repo)
        else:
            model_id = litert_models.add(args.repo, args.file)
    except ValueError as e:
        print(e)
        return 1
    print(f"Added {model_id}. Pick it in the chat page's model menu, or ask for model \"{model_id}\".")


def models_remove(args):
    from . import litert_models
    try:
        freed = litert_models.remove(args.model)
    except LookupError as e:
        print(e)
        return 1
    print(f"Removed {args.model}, freeing {gigabytes(freed)}.")


def users_list(args):
    from .users import Users
    for user in Users().list():
        role = "owner" if user["owner"] else "admin" if user["admin"] else "user"
        key = "has a key" if user["has_key"] else "no key: this PC only" if user["owner"] else "no key"
        print(f"{user['name']:<24} {role:<6} {key}")


def users_change(args):
    from .users import Users
    users = Users()
    try:
        if args.users_command == "add":
            key = users.add(args.name, args.admin)
        elif args.users_command == "key":
            key = users.new_key(args.name)
        elif args.users_command == "admin":
            users.set_admin(args.name, not args.off)
            print(f"{args.name} is {'no longer ' if args.off else ''}an admin.")
            return 0
        else:
            users.remove(args.name)
            print(f"Removed {args.name}; their key no longer works.")
            return 0
    except ValueError as e:
        print(e)
        return 1
    print(f"{args.name}'s key, which can't be shown again:\n\n    {key}\n\n"
          f"Send it as \"Authorization: Bearer <key>\", or give it to the chat page when it asks.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lite-assistant",
                                     description="Local OpenAI-compatible speech, transcription and Gemma chat server")
    commands = parser.add_subparsers(dest="command", required=True)
    serve_parser = commands.add_parser("serve", help="Run the server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Address to listen on (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    serve_parser.add_argument("--tools-dir", action="append", default=[], metavar="FOLDER",
                              help="A folder of Python files whose tools(engines) functions return more built-in "
                                   "tools, besides ~/.config/lite-assistant/tools; repeatable. See docs/guides/write-a-tool-file.md")
    serve_parser.add_argument("--behind-proxy", action="store_true",
                              help="Require a key from every request, even from this PC, as when an https "
                                   "proxy on this PC forwards requests from elsewhere")
    serve_parser.add_argument("--allow-public", action="store_true",
                              help="Take requests with a key from public addresses too, not only this PC, "
                                   "the local network and Tailscale, as for a server in the cloud")
    serve_parser.add_argument("--certfile", metavar="PEM",
                              help="Serve https with this certificate, such as from tailscale cert or mkcert; "
                                   "needs --keyfile")
    serve_parser.add_argument("--keyfile", metavar="PEM", help="The certificate's private key")
    serve_parser.add_argument("--forget-chats-after", type=float, metavar="DAYS",
                              help="Delete chats unchanged for this many days, but pinned ones (default: keep them)")
    serve_parser.add_argument("--llm", choices=LLM_CHOICES,
                              help="The chat model to load: E2B, E4B, or none, to serve voice and the rest "
                                   "without one, which leaves its memory to other work. Without it, the "
                                   "answer 'setup --llm' kept, or else the backend's own")
    serve_parser.set_defaults(run=serve)
    users_parser = commands.add_parser("users", help="List, add and remove users, and give them keys")
    users_parser.set_defaults(run=users_list)
    user_commands = users_parser.add_subparsers(dest="users_command")
    user_commands.add_parser("list", help="List the users (the default)").set_defaults(run=users_list)
    user_add = user_commands.add_parser("add", help="Add a user, and show their key")
    user_add.add_argument("name", help="1 to 32 lowercase letters, digits, - and _")
    user_add.add_argument("--admin", action="store_true", help="Let them add and remove models, and use helpers")
    user_key = user_commands.add_parser("key", help="Give a user a new key; the old one stops working")
    user_key.add_argument("name")
    user_admin = user_commands.add_parser("admin", help="Make a user an admin, or with --off, not")
    user_admin.add_argument("name")
    user_admin.add_argument("--off", action="store_true")
    user_remove = user_commands.add_parser("remove", help="Remove a user; their key stops working")
    user_remove.add_argument("name")
    for command in (user_add, user_key, user_admin, user_remove):
        command.set_defaults(run=users_change)
    setup_parser = commands.add_parser("setup", help="Download the models the engines need")
    setup_parser.add_argument("--llm", choices=LLM_CHOICES,
                              help="The chat model to download: Gemma 4 E2B (2.6 GB), E4B (3.7 GB), or none, for "
                                   "boards only. Without it, setup asks once in a terminal; from a script, it "
                                   "downloads E2B, and E4B too if a GPU works")
    setup_parser.set_defaults(run=setup)
    models_parser = commands.add_parser("models", help="List, find, add and remove LiteRT-LM chat models")
    models_parser.set_defaults(run=models_list)
    model_commands = models_parser.add_subparsers(dest="models_command")
    model_commands.add_parser("list", help="List the downloaded models (the default)").set_defaults(run=models_list)
    search_parser = model_commands.add_parser("search", help="Find LiteRT-LM chat models on Hugging Face")
    search_parser.add_argument("text", nargs="*", help="Words in the model's name, e.g. qwen3")
    search_parser.add_argument("--limit", type=int, default=30, help="Repos to look through (default: 30)")
    search_parser.add_argument("--sort", choices=["downloads", "likes", "trending", "recent"], default="downloads",
                               help="The order to look through them in (default: downloads)")
    search_parser.set_defaults(run=models_search)
    add_parser = model_commands.add_parser("add", help="Download a model from Hugging Face, or copy in a local file")
    add_parser.add_argument("repo", help="The Hugging Face repo, e.g. litert-community/Qwen3-0.6B, its "
                                         "huggingface.co address, or a local .litertlm file")
    add_parser.add_argument("file", nargs="?", help="The .litertlm file, if not the one search shows")
    add_parser.set_defaults(run=models_add)
    remove_parser = model_commands.add_parser("remove", help="Delete a downloaded model")
    remove_parser.add_argument("model", help="The model's id, as 'lite-assistant models' lists it")
    remove_parser.set_defaults(run=models_remove)
    args = parser.parse_args(argv)
    return args.run(args) or 0


if __name__ == "__main__":
    sys.exit(main())
