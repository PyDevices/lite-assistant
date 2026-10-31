"""
Who is asking: the users, each with an API key, kept in users.json in the
data directory. A request from this PC needs no key and comes from the
owner, an admin made on first run and named after the account the server
runs as. Any other request needs a user's key, sent as OpenAI's are:
"Authorization: Bearer <key>". So with no users added, only this PC can
use the server, wherever it listens. And unless allowed, only private
addresses can use it at all, key or not: the home network, and Tailscale's.

Only a hash of each key is kept, so the file gives no keys away. The file
is read again when it changes, so users added from the command line work
without a restart.
"""
import getpass
import hashlib
import ipaddress
import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from . import paths

# A name is also the name of the user's folder, so it's kept to safe characters
NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
KEY_PREFIX = "la-"


@dataclass(frozen=True)
class User:
    name: str
    admin: bool = False

    def folder(self, *parts: str) -> str:
        """ Where the user's own files go, such as their notes and skills. """
        return paths.data_dir("users", self.name, *parts)


def digest(key: str) -> str:
    """
    A key's hash. Keys are long and random, so a plain SHA-256 is enough;
    and looking a hash up, not comparing keys, gives away nothing by its timing.
    """
    return hashlib.sha256(key.encode()).hexdigest()


def account_name() -> str:
    """ The owner's name on first run: the account's, made a valid name. """
    try:
        name = re.sub(r"[^a-z0-9_-]", "", getpass.getuser().lower())
    except Exception:
        name = ""
    return name[:32] if NAME.fullmatch(name[:32]) else "owner"


class Users:
    def __init__(self, path: str = ""):
        self.path = path or paths.data_dir("users.json")
        self.lock = threading.Lock()
        self.read_at: Optional[float] = None
        self.data: Dict = {}
        self.by_hash: Dict[str, str] = {}

    def fresh(self) -> Dict:
        """ The file's contents, read again if it has changed; a missing file is an owner with no key. """
        try:
            changed = os.path.getmtime(self.path)
        except OSError:
            changed = None
        if changed != self.read_at or not self.data:
            data = {}
            if changed is not None:
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
            if not data.get("owner"):
                owner = account_name()
                data = {"owner": owner, "users": {owner: {"admin": True, "created": int(time.time())},
                                                  **data.get("users", {})}}
            self.data, self.read_at = data, changed
            self.by_hash = {about["key"]: name for name, about in data["users"].items() if about.get("key")}
        return self.data

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        temp = self.path + ".tmp"
        # Readable by the account the server runs as only
        with open(os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(temp, self.path)
        self.read_at = os.path.getmtime(self.path)
        self.by_hash = {about["key"]: name for name, about in self.data["users"].items() if about.get("key")}

    def owner(self) -> User:
        with self.lock:
            name = self.fresh()["owner"]
        return User(name, True)

    def by_key(self, key: str) -> Optional[User]:
        with self.lock:
            data = self.fresh()
            name = self.by_hash.get(digest(key))
            return User(name, bool(data["users"][name].get("admin"))) if name else None

    def get(self, name: str) -> Optional[User]:
        with self.lock:
            about = self.fresh()["users"].get(name)
        return User(name, bool(about.get("admin"))) if about is not None else None

    def list(self) -> List[Dict]:
        """ Each user, with whether they're the owner, an admin, and have a key. """
        with self.lock:
            data = self.fresh()
            return [{"name": name, "owner": name == data["owner"], "admin": bool(about.get("admin")),
                     "has_key": bool(about.get("key")), "created": about.get("created")}
                    for name, about in sorted(data["users"].items())]

    def add(self, name: str, admin: bool = False) -> str:
        """ Adds a user, and gives their new key, which isn't kept and can't be shown again. """
        if not NAME.fullmatch(name):
            raise ValueError("A name is 1 to 32 lowercase letters, digits, - and _, starting with a letter or digit.")
        with self.lock:
            if name in self.fresh()["users"]:
                raise ValueError(f"There's already a user called {name}.")
            self.data["users"][name] = {"admin": admin, "created": int(time.time())}
            return self.rekey(name)

    def new_key(self, name: str) -> str:
        """ Gives a user a new key, so the old one stops working. """
        with self.lock:
            if name not in self.fresh()["users"]:
                raise ValueError(f"There's no user called {name}.")
            return self.rekey(name)

    def rekey(self, name: str) -> str:
        key = KEY_PREFIX + secrets.token_urlsafe(32)
        self.data["users"][name]["key"] = digest(key)
        self.save()
        return key

    def set_admin(self, name: str, admin: bool) -> None:
        with self.lock:
            data = self.fresh()
            if name not in data["users"]:
                raise ValueError(f"There's no user called {name}.")
            if name == data["owner"] and not admin:
                raise ValueError("The owner is always an admin.")
            data["users"][name]["admin"] = admin
            self.save()

    def remove(self, name: str) -> None:
        """ Removes a user, so their key stops working. """
        with self.lock:
            data = self.fresh()
            if name not in data["users"]:
                raise ValueError(f"There's no user called {name}.")
            if name == data["owner"]:
                raise ValueError("The owner can't be removed.")
            del data["users"][name]
            self.save()


# --- Checking requests ------------------------------------------------------

# The chat page itself holds no secrets, so anyone may load it; it asks for a key
PUBLIC = {"/", "/favicon.ico", "/llms.txt", "/v1", "/v1/"}
# A browser can't set headers on a WebSocket, so it sends its key as a subprotocol, as OpenAI's Realtime API takes it
KEY_PROTOCOL = "openai-insecure-api-key."
# Failed keys from one address before it must wait, and how long, in seconds
FAILURES, LOCKOUT = 10, 60.0


# Shared address space, which Tailscale gives its devices; Python doesn't count it as private
TAILSCALE = ipaddress.ip_network("100.64.0.0/10")


def is_private(host: str) -> bool:
    """ Whether an address is this PC's, the local network's or Tailscale's, not the internet's. """
    try:
        ip = ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        return False
    ip = getattr(ip, "ipv4_mapped", None) or ip
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip in TAILSCALE


def is_local(host: str) -> bool:
    return host == "::1" or host.startswith("127.") or host.startswith("::ffff:127.")


def sent_key(scope) -> Optional[str]:
    """ The key a request sends: as a Bearer token, or a WebSocket subprotocol. """
    for name, value in scope.get("headers", []):
        if name == b"authorization":
            scheme, _, key = value.decode("latin-1").partition(" ")
            if scheme.lower() == "bearer" and key.strip():
                return key.strip()
    for protocol in scope.get("subprotocols", []):
        if protocol.startswith(KEY_PROTOCOL):
            return protocol[len(KEY_PROTOCOL):]
    return None


class Auth:
    """
    ASGI middleware that finds each request's user, from its key or, for a
    request from this PC, the owner, and puts it in the scope, where an
    endpoint reads it as request.user. A request with no user is refused,
    except for the chat page. Behind a proxy on this PC, every request
    arrives from this PC, so behind_proxy makes every request need a key.
    A request from a public address is refused, even with a key, unless
    allow_public, as on a server in the cloud.
    """
    def __init__(self, app, users: Users, behind_proxy: bool = False, allow_public: bool = False):
        self.app, self.users = app, users
        self.behind_proxy, self.allow_public = behind_proxy, allow_public
        # Each address's recent failed keys, as times
        self.failures: Dict[str, List[float]] = {}

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        host = (scope.get("client") or ("",))[0] or ""
        if not self.allow_public and not is_private(host):
            scope["user"] = None
            return await self.refuse(scope, send, 403, "This server takes requests only from private addresses; "
                                     "its owner can allow others with 'serve --allow-public'.", "public_address")
        key, user, problem = sent_key(scope), None, None
        local = not self.behind_proxy and is_local(host)
        # From this PC, a key picks its user; one that isn't anyone's, such as the
        # placeholder the openai client needs, still means the owner
        if key and local:
            user = self.users.by_key(key) or self.users.owner()
        elif key:
            now = time.monotonic()
            recent = [t for t in self.failures.get(host, []) if now - t < LOCKOUT]
            if len(recent) >= FAILURES:
                problem = (429, "Too many wrong API keys; wait a minute and try again.", "rate_limit_exceeded")
            else:
                user = self.users.by_key(key)
                if user is None:
                    self.failures[host] = recent + [now]
                    problem = (401, "Incorrect API key provided.", "invalid_api_key")
        elif local:
            user = self.users.owner()
        else:
            problem = (401, "You didn't provide an API key. Send it as \"Authorization: Bearer <key>\"; "
                            "the server's owner makes one with 'lite-assistant users add'.", "missing_api_key")
        scope["user"] = user
        if problem and not (scope["type"] == "http" and scope["path"] in PUBLIC):
            return await self.refuse(scope, send, *problem)
        await self.app(scope, receive, send)

    @staticmethod
    async def refuse(scope, send, status: int, message: str, code: str):
        if scope["type"] == "websocket":
            # Closing before accepting refuses the handshake
            return await send({"type": "websocket.close", "code": 1008, "reason": message[:120]})
        body = json.dumps({"error": {"message": message, "type": "invalid_request_error", "code": code}}).encode()
        headers = [(b"content-type", b"application/json")] + ([(b"www-authenticate", b"Bearer")] if status == 401 else [])
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})


def admin_only(request) -> None:
    """ Refuses a request from a user who isn't an admin. """
    from .api_common import APIError
    if not request.user or not request.user.admin:
        raise APIError(403, "Only an admin can do this.", code="admin_only")
