"""
search_notes: the user's notes, .txt and .md files in their notes folder, and documents
there (PDFs, Word files and, with an OCR engine, photos of pages), searched by
meaning with the embedding engine, then put in order by the reranker if
there is one. Each paragraph is embedded once, and again only when its file
changes.
"""
import os
import threading
from typing import Any, Dict, List, Tuple

from .. import paths
from ..documents import DOCUMENTS, PHOTOS, document_text

# The most paragraphs a search returns, and the most characters of each that are embedded
RESULTS = 3
MAX_PARAGRAPH = 2000
# How many of the closest paragraphs by meaning the reranker puts in order
CANDIDATES = 8


def paragraphs(text: str) -> List[str]:
    return [p.strip()[:MAX_PARAGRAPH] for p in text.split("\n\n") if p.strip()]


class Notes:
    def __init__(self, embedder, folder: str = "", ocr=None, reranker=None):
        self.embedder = embedder
        self.folder = folder or paths.data_dir("notes")
        self.ocr = ocr
        self.reranker = reranker
        # path -> (modified time, [(paragraph, vector)])
        self.cache: Dict[str, Tuple[float, List[Tuple[str, List[float]]]]] = {}
        self.lock = threading.Lock()

    def indexed(self, path: str) -> List[Tuple[str, str, List[float]]]:
        """
        The file's paragraphs, with where in it each is (its page, in a PDF)
        and their vectors, embedded with the document task prefix. A file
        that can't be read has none, until it changes.
        """
        modified = os.path.getmtime(path)
        cached = self.cache.get(path)
        if cached and cached[0] == modified:
            return cached[1]
        try:
            parts = document_text(path, self.ocr)
        except Exception as e:
            print(f"[Notes] Can't read {path}: {type(e).__name__}: {e}")
            parts = []
        title = os.path.splitext(os.path.basename(path))[0]
        index = [(where, p, self.embedder.embed(f"title: {title} | text: {p}"))
                 for where, text in parts for p in paragraphs(text)]
        self.cache[path] = (modified, index)
        return index

    def files(self, folder: str) -> List[str]:
        """ The notes and documents in the folder, photos only if they can be read. """
        kinds = (".txt", ".md") + DOCUMENTS + (PHOTOS if self.ocr else ())
        return sorted(os.path.join(root, name) for root, _, names in os.walk(folder)
                      for name in names if name.lower().endswith(kinds))

    def search_notes(self, query: str, user=None) -> Dict[str, Any]:
        """
        Searches the user's own notes and documents by meaning, such as something they wrote down, or
        what a manual, letter or report of theirs says.

        Args:
            query: What to look for in the notes.
        """
        # Each user has their own notes; the server's default folder is for no one in particular
        folder = user.folder("notes") if user else self.folder
        files = self.files(folder)
        if not files:
            return {"results": [], "note": f"There are no notes yet. They go in {folder} as .txt, .md, "
                                           ".pdf or .docx files."}
        with self.lock:
            # Forget the folder's deleted files
            inside = os.path.join(folder, "")
            self.cache = {path: entry for path, entry in self.cache.items()
                          if path in files or not path.startswith(inside)}
            found = [(", ".join(filter(None, (os.path.relpath(path, folder), where))), p, vector)
                     for path in files for where, p, vector in self.indexed(path)]
        query_vector = self.embedder.embed(f"task: search result | query: {query}")
        # The vectors are unit length, so the dot product is the cosine similarity
        scored = sorted(((sum(a * b for a, b in zip(query_vector, vector)), note, p) for note, p, vector in found),
                        reverse=True)
        if self.reranker and len(scored) > 1:
            candidates = scored[:CANDIDATES]
            relevance = self.reranker.rank(query, [p for _, _, p in candidates])
            scored = sorted(((r, note, p) for r, (_, note, p) in zip(relevance, candidates)), reverse=True)
        return {"results": [{"note": note, "text": p, "score": round(score, 3)}
                            for score, note, p in scored[:RESULTS]]}
