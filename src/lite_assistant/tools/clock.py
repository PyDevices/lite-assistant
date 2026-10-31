"""
The clock tools: the time, and timers. A timer can't ring, since the server
can't speak unprompted, so asking about the timers is how to hear one is done.
"""
import threading
import time
from datetime import datetime
from typing import Any, Dict, List


def get_time() -> Dict[str, str]:
    """ The current local date, time and time zone. """
    now = datetime.now().astimezone()
    return {"date": f"{now:%A}, {now:%B} {now.day}, {now.year}", "time": f"{now:%H:%M}",
            "time_12h": f"{now.hour % 12 or 12}:{now:%M %p}", "timezone": now.tzname() or "",
            "utc_offset": f"{now:%z}"}


def remaining(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return " ".join(f"{n} {unit}{'s' if n != 1 else ''}" for n, unit in
                    ((hours, "hour"), (minutes, "minute"), (seconds, "second")) if n) or "0 seconds"


class Timers:
    """ Timers kept in memory, so a restart forgets them. """
    def __init__(self):
        self.timers: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def set_timer(self, seconds: int, label: str = "") -> Dict[str, Any]:
        """
        Sets a timer.

        Args:
            seconds: How long the timer runs, in seconds.
            label: What the timer is for, if the user said.
        """
        seconds = float(seconds)
        if seconds <= 0:
            raise ValueError("a timer needs a positive number of seconds")
        with self.lock:
            self.timers.append({"label": label or f"{remaining(seconds)} timer", "ends": time.time() + seconds})
        return {"set": label or "timer", "duration": remaining(seconds)}

    def list_timers(self) -> Dict[str, Any]:
        """ Lists the timers that are set, with the time left on each, and any that have finished. """
        now = time.time()
        with self.lock:
            # A finished timer is listed once, then forgotten
            self.timers = [t for t in self.timers if not t.get("reported")]
            listed = []
            for timer in self.timers:
                left = timer["ends"] - now
                listed.append({"label": timer["label"], "left": remaining(left)} if left > 0
                              else {"label": timer["label"], "finished": f"{remaining(-left)} ago"})
                timer["reported"] = left <= 0
        return {"timers": listed} if listed else {"timers": [], "note": "No timers are set."}
