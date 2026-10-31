"""
Engines that load on first use and unload when idle, so the server can have
many models without holding them all in memory.
"""
import gc
import os
import threading
import time
from typing import Optional

# An engine unused this long, in seconds, is unloaded; 0 keeps engines loaded
IDLE = float(os.environ.get("LITE_ASSISTANT_IDLE_UNLOAD", 600))


class LazyEngine:
    """
    Stands in for an engine of class cls, which it creates the first time
    anything but its model id is asked of it. unload_idle drops the engine
    once it's gone unused for idle seconds; the next use loads it again. A
    call already under way keeps its engine until it returns.
    """
    def __init__(self, cls: type, idle: Optional[float] = None):
        self._cls, self._idle = cls, IDLE if idle is None else idle
        self._engine, self._used = None, 0.0
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def _get(self):
        with self._lock:
            if self._engine is None:
                self._engine = self._cls()
            self._used = time.monotonic()
            return self._engine

    def get_model_id(self) -> str:
        # Known without loading, for /v1/models, if the class says it
        model_id = getattr(self._cls, "MODEL_ID", None)
        return model_id if model_id and self._engine is None else self._get().get_model_id()

    def __getattr__(self, name):
        if self._engine is None:
            # The class's constants, such as what it does (TASK), are known without loading it
            constant = getattr(self._cls, name, None)
            if constant is not None and not callable(constant) and not isinstance(constant, property):
                return constant
            # What only a loaded engine has, such as its backend, isn't worth loading it for
            if name == "backend":
                return None
        return getattr(self._get(), name)

    def unload_idle(self, now: Optional[float] = None) -> bool:
        """ Unloads the engine if it's gone unused for long enough; True if it did. """
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._engine is None or not self._idle or now - self._used < self._idle:
                return False
            print(f"[Engines] Unloading {self._cls.__name__}, unused for {now - self._used:.0f}s.")
            self._engine = None
        gc.collect()
        return True

    def __repr__(self):
        return f"LazyEngine({self._cls.__name__}, {'loaded' if self.loaded else 'not loaded'})"
