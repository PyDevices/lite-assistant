"""
The server's memory of responses and conversations, so a client can
continue from an earlier reply, or a conversation, by id instead of
resending the history.

Conversations are kept in a SQLite database, which the server keeps in the
data directory, so they outlast a restart; each is its user's alone. With
no database file, as in tests, they're in memory. With one, the photos,
voice clips and documents in a conversation's items are kept as its
user's media files (see files.py), and deleted with it. A conversation's
updated_at is when it last changed, which the chat list orders by, and
which forgetting old conversations counts from; one whose metadata has
"pinned" is never forgotten.

Responses are in memory only: a restart forgets them, and the oldest are
dropped past a cap.
"""
import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from .files import Files, to_files
from .users import User

MAX_RESPONSES = 1000
# How often, in seconds, old conversations are looked for, when they're forgotten
TIDY_EVERY = 3600
# The most items a conversation can be created with, or have added at once
MAX_ITEMS_PER_ADD = 20

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    metadata TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS conversations_by_user ON conversations (user, updated_at);
CREATE TABLE IF NOT EXISTS items (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation TEXT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    id TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS items_by_conversation ON items (conversation, seq);
"""


class StoredResponse:
    """ A response object, the input items it was given, and the context it was written in. """
    def __init__(self, response: Dict[str, Any], input_items: List[Dict[str, Any]],
                 context: List[Dict[str, Any]], user: str = ""):
        self.response = response
        # Who made it: no one else sees it
        self.user = user
        self.input_items = input_items
        # Every item the model saw, earlier turns included, except instructions,
        # which OpenAI doesn't carry over to a chained response
        self.context = context
        # Set to stop a background response mid-reply
        self.cancel = threading.Event()

    def chained_context(self) -> List[Dict[str, Any]]:
        """ The items a response chained onto this one starts from: its context and its reply. """
        return self.context + self.response["output"]


class Conversation:
    """ A conversation object and its items, oldest first, as they were when it was read. """
    def __init__(self, conversation_id: str, metadata: Optional[Dict[str, str]], user: str = "",
                 created_at: Optional[int] = None, updated_at: Optional[int] = None):
        self.user = user
        self.object = {"id": conversation_id, "object": "conversation",
                       "created_at": int(time.time()) if created_at is None else created_at,
                       "metadata": metadata or {}}
        self.updated_at = self.object["created_at"] if updated_at is None else updated_at
        self.items: List[Dict[str, Any]] = []

    @property
    def id(self) -> str:
        return self.object["id"]

    def item(self, item_id: str) -> Optional[Dict[str, Any]]:
        return next((item for item in self.items if item["id"] == item_id), None)


class Store:
    def __init__(self, database: str = "", forget_after: Optional[float] = None, max_responses: int = MAX_RESPONSES):
        """ forget_after is how many days unchanged a conversation is kept, or None to keep them all. """
        self.lock = threading.RLock()
        self.forget_after, self.tidied = forget_after, 0.0
        # Media goes in files only when conversations are kept in a file too
        self.media = bool(database)
        self.max_responses = max_responses
        self.responses: "OrderedDict[str, StoredResponse]" = OrderedDict()
        if database:
            os.makedirs(os.path.dirname(os.path.abspath(database)), exist_ok=True)
        self.db = sqlite3.connect(database or ":memory:", check_same_thread=False, isolation_level=None)
        if database:
            # Readers don't wait for a writer, and a crash mid-write loses nothing written before
            self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self.tidy()

    # --- Responses ---

    def put_response(self, stored: StoredResponse):
        with self.lock:
            self.responses[stored.response["id"]] = stored
            while len(self.responses) > self.max_responses:
                self.responses.popitem(last=False)

    def get_response(self, response_id: str) -> Optional[StoredResponse]:
        with self.lock:
            return self.responses.get(response_id)

    def delete_response(self, response_id: str) -> bool:
        with self.lock:
            return self.responses.pop(response_id, None) is not None

    # --- Conversations ---

    def put_conversation(self, conversation: Conversation):
        """ Saves a new conversation, with its items. """
        self.tidy()
        with self.lock, self.db:
            self.db.execute("BEGIN")
            self.db.execute("INSERT INTO conversations VALUES (?, ?, ?, ?, ?)",
                            (conversation.id, conversation.user, conversation.object["created_at"],
                             conversation.updated_at, json.dumps(conversation.object["metadata"])))
            self.insert_items(conversation, conversation.items)

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        with self.lock:
            row = self.db.execute("SELECT user, created_at, updated_at, metadata FROM conversations WHERE id = ?",
                                  (conversation_id,)).fetchone()
            if not row:
                return None
            conversation = Conversation(conversation_id, json.loads(row[3]), row[0], row[1], row[2])
            conversation.items = [json.loads(data) for data, in self.db.execute(
                "SELECT data FROM items WHERE conversation = ? ORDER BY seq", (conversation_id,))]
            return conversation

    def list_conversations(self, user: str) -> List[Conversation]:
        """ A user's conversations, without their items, the last changed first. """
        with self.lock:
            rows = self.db.execute("SELECT id, created_at, updated_at, metadata FROM conversations WHERE user = ? "
                                   "ORDER BY updated_at DESC, rowid DESC", (user,)).fetchall()
        return [Conversation(id, json.loads(metadata), user, created, updated)
                for id, created, updated, metadata in rows]

    def add_items(self, conversation: Conversation, items: List[Dict[str, Any]]):
        with self.lock, self.db:
            self.db.execute("BEGIN")
            self.insert_items(conversation, items)
            self.touch(conversation)
        conversation.items.extend(items)

    def delete_item(self, conversation: Conversation, item_id: str):
        with self.lock, self.db:
            self.db.execute("BEGIN")
            self.db.execute("DELETE FROM items WHERE conversation = ? AND id = ?", (conversation.id, item_id))
            self.touch(conversation)
        conversation.items = [item for item in conversation.items if item["id"] != item_id]

    def set_metadata(self, conversation: Conversation, metadata: Dict[str, str]):
        with self.lock, self.db:
            self.db.execute("BEGIN")
            self.db.execute("UPDATE conversations SET metadata = ? WHERE id = ?",
                            (json.dumps(metadata), conversation.id))
            self.touch(conversation)
        conversation.object = {**conversation.object, "metadata": metadata}

    def delete_conversation(self, conversation_id: str) -> bool:
        with self.lock, self.db:
            row = self.db.execute("SELECT user FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if not row:
                return False
            self.db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        if self.media:
            Files(User(row[0])).delete_folder(conversation_id)
        return True

    def forget(self, days: float) -> int:
        """ Deletes the conversations, but pinned ones, unchanged for days; how many. """
        before = time.time() - days * 86400
        with self.lock:
            old = self.db.execute("SELECT id FROM conversations WHERE updated_at < ? AND "
                                  "coalesce(json_extract(metadata, '$.pinned'), '') IN ('', 'false', '0')",
                                  (before,)).fetchall()
            return sum(self.delete_conversation(conversation_id) for conversation_id, in old)

    def tidy(self):
        """ Forgets old conversations, if they're forgotten, at most every TIDY_EVERY seconds. """
        if self.forget_after is None or time.time() - self.tidied < TIDY_EVERY:
            return
        self.tidied = time.time()
        forgotten = self.forget(self.forget_after)
        if forgotten:
            print(f"Forgot {forgotten} conversation(s) unchanged for {self.forget_after:g} days")

    def insert_items(self, conversation: Conversation, items: List[Dict[str, Any]]):
        """ Saves items, with their media as files in a folder of the conversation's. """
        if self.media:
            items = to_files(items, Files(User(conversation.user)), conversation.id)
        self.db.executemany("INSERT INTO items (conversation, id, data) VALUES (?, ?, ?)",
                            [(conversation.id, item["id"], json.dumps(item)) for item in items])

    def touch(self, conversation: Conversation):
        conversation.updated_at = int(time.time())
        self.db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                        (conversation.updated_at, conversation.id))


def page(items: List[Dict[str, Any]], after: Optional[str], limit: int, order: str) -> Dict[str, Any]:
    """ One page of a cursor-paginated list, as OpenAI's list endpoints return it. """
    items = list(reversed(items)) if order == "desc" else list(items)
    if after:
        ids = [item["id"] for item in items]
        items = items[ids.index(after) + 1:] if after in ids else []
    data = items[:limit]
    return {"object": "list", "data": data, "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None, "has_more": len(items) > limit}
