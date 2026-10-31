"""
LiteRT-LM chat models: those downloaded to the Hugging Face cache or put in
the models folder of the data directory, which the LiteRT-LM engine can
switch between, and those on the Hub that can be added.
A model's id is its file name without .litertlm, e.g. "gemma-4-E2B-it" or
"Qwen3-0.6B".
"""
import concurrent.futures
import contextlib
import ctypes
import functools
import os
import re
import shutil
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import paths

SUFFIX = ".litertlm"
# The Hub tags LiteRT-LM models with their library's name
LIBRARY = "litert-lm"
# Memory the rest of the server and the OS need beside a model: the TTS, STT,
# embedding and VAD models, and the first steps of its context
RESERVED_MEMORY = 3 * 2**30
# Tags of LiteRT-LM models that aren't chat models, such as speech recognizers
NOT_CHAT = {"automatic-speech-recognition", "text-to-speech", "feature-extraction", "sentence-similarity"}
# Names of those that aren't tagged as such: embedding, speech and speech recognition models, base
# models (Qwen's -Base, Gemma's -pt), which continue text rather than answer it, and random-weight tests
NOT_CHAT_NAME = re.compile(r"embedding|(?<![a-z])(tts|asr|whisper|base|random)(?![a-z])|-pt(?=$|[-_.]lite)",
                           re.IGNORECASE)
# How the Hub's models can be sorted, by the name the server takes, to the name the Hub does
SORTS = {"downloads": "downloads", "likes": "likes", "trending": "trending_score", "recent": "last_modified"}
# A Hub address: huggingface.co/owner/repo, then /resolve/REV/FILE, /blob/REV/FILE or /tree/REV
HUB_URL = re.compile(r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/([\w.-]+/[\w.-]+)"
                     r"(?:/(?:resolve|blob)/[^/]+/(.+?)|/tree/[^/]+)?/?(?:[?#].*)?$")
# Files built for one kind of device: an NPU, a browser, or the GPU backend alone
DEVICE_FILE = re.compile(r"mediatek|qualcomm|\bmt\d{4}|sm\d{4}|npu|tpu|[-_.]web\b|[-_.]gpu\b", re.IGNORECASE)


@functools.lru_cache(maxsize=None)
def facts(path: str) -> Optional[Dict[str, Any]]:
    """
    What the model file at path says about itself: the inputs it takes
    ("text", "image", "audio"), its context length (0 if it doesn't say),
    whether it carries a speculative decoding drafter, and whether it says it
    thinks and calls tools (see model_facts). None if it isn't a
    chat model, such as an embedding model, or can't be read.
    """
    import litert_lm
    try:
        with litert_lm.ModelInfo(path) as info:
            if not info.is_llm_model or not info.llm:
                return None
            inputs = info.input_modalities
            return {"modalities": [name for name, takes in (("text", inputs.text), ("image", inputs.vision),
                                                             ("audio", inputs.audio)) if takes],
                    "context": info.max_context_tokens,
                    "speculative": info.llm.has_speculative_decoding_support(),
                    "thinking": bool(info.llm.supports_thinking), "tools": bool(info.llm.supports_function_calling)}
    except Exception:
        return None


def local_folder() -> str:
    """ The folder for .litertlm files that don't come from the Hub, or were copied from elsewhere. """
    return paths.data_dir("models")


def reference(text: str) -> Tuple[str, Optional[str]]:
    """
    The repo and file, if it names one, that a Hub address or "owner/repo"
    or "owner/repo/file.litertlm" points to. Raises ValueError for anything else.
    """
    text = text.strip()
    found = HUB_URL.match(text)
    if found:
        return found.group(1), found.group(2)
    parts = text.strip("/").split("/")
    if len(parts) >= 2 and all(re.fullmatch(r"[\w.-]+", part) for part in parts[:2]):
        return "/".join(parts[:2]), "/".join(parts[2:]) or None
    raise ValueError(f"{text!r} isn't a Hugging Face repo; give it as owner/repo, or its huggingface.co address.")


def downloaded() -> Dict[str, str]:
    """
    Every LiteRT-LM chat model in the Hugging Face cache and the models
    folder, by id, to its path. One in the folder wins over one of the same name in the cache.
    """
    from huggingface_hub import scan_cache_dir
    try:
        repos = scan_cache_dir().repos
    except Exception:
        # No cache yet
        repos = []
    found = {}
    for repo in repos:
        if repo.repo_type != "model":
            continue
        # The newest revision's file wins when several hold one of that name
        for revision in sorted(repo.revisions, key=lambda r: r.last_modified):
            for file in revision.files:
                if file.file_name.endswith(SUFFIX) and facts(str(file.file_path)):
                    found[file.file_name[:-len(SUFFIX)]] = str(file.file_path)
    folder = local_folder()
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        path = os.path.join(folder, name)
        if name.endswith(SUFFIX) and facts(path):
            found[name[:-len(SUFFIX)]] = path
    return dict(sorted(found.items(), key=lambda item: item[0].lower()))


def total_memory() -> Optional[int]:
    """ This machine's RAM in bytes, if the OS says. """
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                            ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                            ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
            status = MemoryStatus(length=ctypes.sizeof(MemoryStatus))
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def fits(size: Optional[int]) -> Optional[bool]:
    """ Whether a model file of size bytes fits in this machine's RAM beside the rest of the server. """
    memory = total_memory()
    return None if memory is None or size is None else size <= memory - RESERVED_MEMORY


def pick_file(repo_id: str, files: Dict[str, Optional[int]]) -> Optional[str]:
    """
    The file to download from a repo's files (name to size): the .litertlm
    named after the repo, which is its full build, or else the smallest,
    leaving out files built for one kind of device. None if the repo has no such file.
    """
    usable = {name: size for name, size in files.items()
              if name.endswith(SUFFIX) and "/" not in name and not DEVICE_FILE.search(name)}
    # Names are matched whatever their case, and with - and _ alike: Qwen3-1.7B holds Qwen3_1.7B.litertlm
    plain = lambda name: name.lower().replace("_", "-")
    named = plain(repo_id.split("/")[-1].removesuffix("-litert-lm").removesuffix("-litertlm") + SUFFIX)
    for name in usable:
        if plain(name) == named:
            return name
    return min(usable, key=lambda name: usable[name] or 0, default=None)


@contextlib.contextmanager
def explained(repo_id: str):
    """ Turns the Hub's errors for a repo that's gated, private or missing into a ValueError saying what to do. """
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    try:
        yield
    except GatedRepoError:
        raise ValueError(f"{repo_id} is gated: accept its terms at https://huggingface.co/{repo_id} while "
                         "logged in, then log this machine in with 'hf auth login' or set HF_TOKEN.")
    except RepositoryNotFoundError:
        raise ValueError(f"There's no repo {repo_id} on Hugging Face, or it's private; if it's yours, log this "
                         "machine in with 'hf auth login' or set HF_TOKEN.")
    except HfHubHTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in (401, 403):
            raise ValueError(f"Hugging Face won't give {repo_id} to this machine (HTTP {status}); log in with "
                             "'hf auth login' or set HF_TOKEN, with a token that can read it.")
        raise


def repo_info(repo_id: str):
    """ What the Hub says about a repo, with its files' sizes. """
    from huggingface_hub import HfApi
    with explained(repo_id):
        return HfApi().model_info(repo_id, files_metadata=True)


def repo_files(repo_id: str) -> Dict[str, Optional[int]]:
    """ A Hub repo's LiteRT-LM files, by name, with their sizes. """
    return files_of(repo_info(repo_id))


def files_of(info) -> Dict[str, Optional[int]]:
    """ The LiteRT-LM files in what the Hub says about a repo, by name, with their sizes. """
    return {f.rfilename: f.size for f in info.siblings or [] if f.rfilename.endswith(SUFFIX)}


def describe(repo, have, file: Optional[str] = None, named: bool = False) -> Optional[Dict[str, Any]]:
    """
    A repo as search lists it, or None if it isn't a chat model: the file
    add() would take from it, or the one named, and its size. A repo asked
    for by name is described whatever it's called, and its errors raised.
    """
    # A repo built for one kind of device may name it only in the repo's name
    if not named and (NOT_CHAT & set(repo.tags or []) or NOT_CHAT_NAME.search(repo.id) or DEVICE_FILE.search(repo.id)):
        return None
    try:
        # A repo asked for by name has been looked up already
        info = repo if named else repo_info(repo.id)
        files = files_of(info)
    except Exception:
        if named:
            raise
        return None
    name = file if file in files else pick_file(repo.id, files)
    if not name:
        if named:
            raise ValueError(f"{repo.id} has no LiteRT-LM file{f' named {file}' if file else ''}.")
        return None
    return {"repo": repo.id, "file": name, "id": name[:-len(SUFFIX)], "size": files[name],
            "fits": fits(files[name]), "downloaded": name in have, "downloads": getattr(repo, "downloads", 0) or 0,
            "likes": getattr(repo, "likes", 0) or 0, "other_files": sorted(set(files) - {name}),
            # Its files are listed for anyone, but only those who've accepted its terms can download them
            "gated": bool(getattr(info, "gated", False))}


def search(query: Optional[str] = None, limit: int = 30, sort: str = "downloads") -> List[Dict[str, Any]]:
    """
    LiteRT-LM models on the Hub, most downloaded first, or by sort (a key of
    SORTS): each one's repo, the file add() would take from it, its size,
    whether it fits in RAM, and whether that file is downloaded already. A
    query that's a Hub address or owner/repo gives that repo alone, or
    raises ValueError if it can't be added.
    """
    from huggingface_hub import HfApi
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {', '.join(SORTS)}")
    have = {os.path.basename(path) for path in downloaded().values()}
    if query and "/" in query:
        repo_id, file = reference(query)
        return [describe(repo_info(repo_id), have, file, named=True)]
    repos = list(HfApi().list_models(filter=LIBRARY, search=query or None, sort=SORTS[sort], limit=limit))
    # A repo's file sizes take a request each
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        found = [model for model in pool.map(lambda repo: describe(repo, have), repos) if model]
    return found


def reporting_progress(progress: Callable[[int, Optional[int]], None]):
    """ A tqdm class for hf_hub_download that, in place of a bar, calls progress(bytes so far, total). """
    from tqdm.auto import tqdm

    class Progress(tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**kwargs, "disable": True})

        def update(self, n=1):
            self.n += n or 0
            progress(self.n, self.total)

    return Progress


def add(repo_id: str, file: Optional[str] = None,
        progress: Optional[Callable[[int, Optional[int]], None]] = None) -> str:
    """
    Downloads a LiteRT-LM model from the Hub: file, or the one pick_file
    chooses. Returns its id. Raises ValueError for a repo or file that isn't
    a LiteRT-LM chat model. progress, if given, is called with the bytes
    downloaded so far and the total; otherwise a bar shows in the terminal.
    """
    from huggingface_hub import hf_hub_download
    repo_id, named = reference(repo_id)
    file = file or named
    files = repo_files(repo_id)
    file = file or pick_file(repo_id, files)
    if not file or file not in files:
        raise ValueError(f"{repo_id} has no LiteRT-LM file{f' named {file}' if file else ''}.")
    with explained(repo_id):
        path = hf_hub_download(repo_id, file, tqdm_class=reporting_progress(progress) if progress else None)
    if not facts(path):
        raise ValueError(f"{file} isn't a chat model LiteRT-LM can load; it's downloaded, but won't be offered.")
    return file[:-len(SUFFIX)]


def add_file(path: str) -> str:
    """
    Copies a .litertlm file into the models folder, and returns its id.
    Raises ValueError for a file that isn't a chat model LiteRT-LM can load.
    """
    path = os.path.abspath(os.path.expanduser(path))
    if not path.endswith(SUFFIX) or not os.path.isfile(path):
        raise ValueError(f"{path} isn't a {SUFFIX} file.")
    if not facts(path):
        raise ValueError(f"{os.path.basename(path)} isn't a chat model LiteRT-LM can load.")
    folder = local_folder()
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, os.path.basename(path))
    if os.path.abspath(target) != path:
        shutil.copyfile(path, target)
    return os.path.basename(path)[:-len(SUFFIX)]


def remove(model_id: str) -> int:
    """ Deletes a downloaded model's file from the Hugging Face cache or the models folder, and returns the bytes freed. """
    from huggingface_hub import scan_cache_dir
    freed = 0
    local = os.path.join(local_folder(), model_id + SUFFIX)
    if os.path.basename(local) == model_id + SUFFIX and os.path.isfile(local):
        freed += os.path.getsize(local)
        os.remove(local)
    for repo in scan_cache_dir().repos:
        for revision in repo.revisions:
            for file in revision.files:
                if file.file_name == model_id + SUFFIX:
                    # The snapshot's entry is a link to the blob that holds the data
                    if file.blob_path.exists():
                        freed += file.size_on_disk
                        file.blob_path.unlink()
                    file.file_path.unlink(missing_ok=True)
        # A repo with nothing else downloaded goes too, so scans don't keep finding it
        if freed and repo.repo_path.is_dir() and not any((repo.repo_path / "blobs").glob("*")):
            shutil.rmtree(repo.repo_path)
    if not freed:
        raise LookupError(f"{model_id} isn't downloaded.")
    return freed
