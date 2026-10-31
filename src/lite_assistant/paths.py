"""
Where lite-assistant keeps files. Models hosted on Hugging Face live in its
shared cache (huggingface_hub decides where; HF_HOME moves it). Everything
else lives in per-user directories, or under LITE_ASSISTANT_HOME if set.
"""
import os
from typing import Optional

import platformdirs

APP_NAME = "lite-assistant"


def data_dir(*parts: str) -> str:
    """ Downloaded files that don't come from Hugging Face, such as Kokoro's. """
    home = os.environ.get("LITE_ASSISTANT_HOME")
    base = home or platformdirs.user_data_dir(APP_NAME, appauthor=False)
    return os.path.join(base, *parts)


def config_dir(*parts: str) -> str:
    """ What the server's owner sets up, such as the tools folder: ~/.config/lite-assistant on Linux. """
    home = os.environ.get("LITE_ASSISTANT_HOME")
    base = os.path.join(home, "config") if home else platformdirs.user_config_dir(APP_NAME, appauthor=False)
    return os.path.join(base, *parts)


def cache_dir(*parts: str) -> str:
    """ Files that can be rebuilt, such as LiteRT-LM's compiled model caches. """
    home = os.environ.get("LITE_ASSISTANT_HOME")
    base = os.path.join(home, "cache") if home else platformdirs.user_cache_dir(APP_NAME, appauthor=False)
    return os.path.join(base, *parts)


def cached_hf_file(repo_id: str, filename: str) -> Optional[str]:
    """ The local path of a Hugging Face file if it's already downloaded, without touching the network. """
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(repo_id, filename)
    return path if isinstance(path, str) else None
