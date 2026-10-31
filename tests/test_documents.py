"""
Documents: the text of PDFs, Word files and photos of pages, search_notes
over them with page numbers and reranking, the Qwen3 reranker's prompt and
score with a fake graph, and the /v1/rerank endpoint.
"""
import io
import os
import re
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from lite_assistant.app import Engines, create_app
from lite_assistant.documents import MAX_ATTACHED, attached_text, docx_text, document_text, pdf_pages, photo_text
from lite_assistant.engines import TextLine
from lite_assistant.engines.qwen3rerank_engine import TOKENS, Qwen3RerankEngine
from lite_assistant.tools.notes import Notes


def pdf(*pages) -> bytes:
    """ A PDF with a page for each text, each line of it a line on the page; None is a blank page. """
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None,
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        lines = (text or "").split("\n") if text else []
        stream = "BT /F1 12 Tf 72 720 Td 14 TL " + " ".join(f"({line}) Tj T*" for line in lines) + " ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {len(objects)} 0 R "
                       "/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = "%PDF-1.4\n", []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n"
    table = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{table}\n%%EOF\n"
    return out.encode("latin-1")


def docx(*paragraphs) -> bytes:
    """ A Word file with these paragraphs. """
    word = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f"<w:p><w:r><w:t>{p[:5]}</w:t></w:r><w:r><w:t>{p[5:]}</w:t></w:r></w:p>" for p in paragraphs)
    made = io.BytesIO()
    with zipfile.ZipFile(made, "w") as archive:
        archive.writestr("word/document.xml", f'<w:document xmlns:w="{word}"><w:body>{body}</w:body></w:document>')
    return made.getvalue()


class FakeOCR:
    """ Reads every photo as two lines close together, then a third after a gap. """
    def read(self, image):
        return [TextLine("Water the basil", 0.9, (0.1, 0.10, 0.6, 0.14)),
                TextLine("each morning.", 0.9, (0.1, 0.15, 0.5, 0.19)),
                TextLine("Tomatoes every two days.", 0.9, (0.1, 0.30, 0.7, 0.34))]


class WordEmbedder:
    """ Embeds text as how often it says each of a few words, scaled to unit length. """
    WORDS = ["basil", "tomatoes", "morning", "warranty", "filter"]

    def embed(self, item, dimensions=None):
        words = re.findall(r"\w+", item.lower().split("text:")[-1].split("query:")[-1])
        vector = [words.count(w) for w in self.WORDS] + [0.1]
        size = sum(v * v for v in vector) ** 0.5
        return [v / size for v in vector]


class FakeReranker:
    """ Ranks a document higher the more often it says the query's last word. """
    MODEL_ID = "fake-reranker"

    def __init__(self):
        self.asked = []

    def get_model_id(self):
        return self.MODEL_ID

    def rank(self, query, documents, instruction=None):
        self.asked.append((query, list(documents), instruction))
        word = query.split()[-1].strip("?").lower()
        return [min(1.0, d.lower().count(word) / 3) for d in documents]


class TestDocuments(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def write(self, name, data: bytes) -> str:
        path = os.path.join(self.folder.name, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_pdf_pages_join_wrapped_lines(self):
        path = self.write("manual.pdf", pdf("Clean the filter\nevery month.\nThe warranty lasts two years.", "Page two."))
        self.assertEqual(pdf_pages(path), ["Clean the filter every month.\nThe warranty lasts two years.", "Page two."])

    def test_a_page_with_no_text_is_read_by_ocr(self):
        path = self.write("scan.pdf", pdf(None))
        self.assertEqual(pdf_pages(path), [""])
        with mock.patch("lite_assistant.documents.scanned_page", return_value="Read by OCR.") as scanned:
            self.assertEqual(pdf_pages(path, FakeOCR()), ["Read by OCR."])
        self.assertEqual(scanned.call_args.args[1], 1)

    def test_docx_paragraphs(self):
        path = self.write("letter.docx", docx("Dear neighbour,", "The fence is fixed."))
        self.assertEqual(docx_text(path), "Dear neighbour,\n\nThe fence is fixed.")

    def test_photo_paragraphs_are_split_at_gaps(self):
        self.assertEqual(photo_text(b"photo", FakeOCR()), "Water the basil\neach morning.\n\nTomatoes every two days.")
        path = self.write("page.jpg", b"photo")
        self.assertEqual(document_text(path), [])
        self.assertEqual(document_text(path, FakeOCR())[0][0], "")

    def test_notes_search_documents_with_their_pages(self):
        self.write("manual.pdf", pdf("Intro.", "Clean the filter\nevery month."))
        self.write("letter.docx", docx("The warranty lasts two years."))
        self.write("broken.pdf", b"not a PDF")
        self.write("page.png", b"photo")
        notes = Notes(WordEmbedder(), self.folder.name)
        self.assertEqual(notes.search_notes("filter")["results"][0], {
            "note": "manual.pdf, page 2", "text": "Clean the filter every month.", "score": 1.0})
        self.assertEqual(notes.search_notes("warranty")["results"][0]["note"], "letter.docx")
        # A photo is a note only with an OCR engine to read it
        self.assertNotIn("page.png", [r["note"] for r in notes.search_notes("basil")["results"]])
        notes = Notes(WordEmbedder(), self.folder.name, ocr=FakeOCR())
        self.assertEqual(notes.search_notes("basil")["results"][0]["note"], "page.png")

    def test_attached_files(self):
        self.assertEqual(attached_text("manual.pdf", pdf("Clean the filter.", "Two years.")),
                         "The attached file manual.pdf:\n\n[page 1]\nClean the filter.\n\n[page 2]\nTwo years.")
        self.assertEqual(attached_text("a.csv", b"bed,plant\n3,basil"), "The attached file a.csv:\n\nbed,plant\n3,basil")
        long = attached_text("long.txt", b"x" * (MAX_ATTACHED + 5))
        self.assertTrue(long.endswith("[The rest of long.txt, 5 more characters, was left out.]"))
        with self.assertRaisesRegex(ValueError, "isn't a document"):
            attached_text("song.mp3", b"ID3\0\0")
        with self.assertRaisesRegex(ValueError, "needs an OCR model"):
            attached_text("page.png", b"not read")
        with self.assertRaisesRegex(ValueError, "couldn't be read"):
            attached_text("broken.pdf", b"not a pdf")

    def test_the_reranker_puts_the_closest_in_order(self):
        with open(os.path.join(self.folder.name, "garden.md"), "w") as f:
            f.write("Basil.\n\nBasil and tomatoes, tomatoes, tomatoes.\n\nBasil, basil, basil and the morning.")
        reranker = FakeReranker()
        notes = Notes(WordEmbedder(), self.folder.name, reranker=reranker)
        results = notes.search_notes("What about tomatoes")["results"]
        self.assertEqual(results[0], {"note": "garden.md", "text": "Basil and tomatoes, tomatoes, tomatoes.",
                                      "score": 1.0})
        self.assertEqual(len(reranker.asked[0][1]), 3)


class FakeGraph:
    """ Says yes at each position by as much as the first number of its embedding there. """
    def __init__(self):
        self.embeds = None

    def __call__(self, inputs_embeds):
        self.embeds = inputs_embeds
        logits = np.zeros((1, TOKENS, 2), np.float32)
        logits[0, :, 1] = inputs_embeds[0, :, 0]
        return {"logits": logits}


class FakeTokenizer:
    """ A token per character, each its character's code. """
    def encode(self, text, add_special_tokens=True):
        return type("Encoding", (), {"ids": [ord(c) for c in text]})()


class FakeQwen3Rerank(Qwen3RerankEngine):
    def __init__(self):
        import threading
        self.runner = FakeGraph()
        self.input = "inputs_embeds"
        self.tokenizer = FakeTokenizer()
        self.prefix, self.suffix = [1, 2], [3]
        # Each token's embedding is its id, in every column, scaled down
        self.table = np.tile((np.arange(200, dtype=np.float32) / 100)[:, None], (1, 1024))
        self.lock = threading.Lock()


class TestQwen3Rerank(unittest.TestCase):
    def test_prompt_cuts_the_document_to_fit(self):
        engine = FakeQwen3Rerank()
        ids = engine.ids("q", "x" * 1000, "i")
        self.assertEqual(len(ids), TOKENS)
        self.assertEqual(ids[:2] + ids[-1:], [1, 2, 3])
        self.assertEqual(bytes(ids[2:-1]).decode()[:28], "<Instruct>: i\n<Query>: q\n<Do")
        with self.assertRaises(ValueError):
            engine.ids("q" * 300, "x", "i")

    def test_the_score_is_yes_at_the_last_real_token(self):
        engine = FakeQwen3Rerank()
        # The last token is the suffix's, 3, so yes - no is 0.03 for every document
        scores = engine.rank("q", ["a", "bb"])
        self.assertAlmostEqual(scores[0], 1 / (1 + np.exp(-0.03)), places=5)
        self.assertEqual(len(scores), 2)
        # Padded on the right with zeros
        self.assertFalse(engine.runner.embeds[0, -1].any())


class TestEndpoint(unittest.TestCase):
    def setUp(self):
        self.reranker = FakeReranker()
        engines = Engines({}, {}, {}, rerank={"fake": self.reranker})
        self.http = TestClient(create_app(engines, None, "none"), client=("127.0.0.1", 50000))

    def test_rerank_orders_and_cuts(self):
        reply = self.http.post("/v1/rerank", json={
            "query": "Which is about basil", "top_n": 2, "instruction": "Find basil",
            "documents": ["tomatoes", {"text": "basil basil basil"}, "basil"]}).json()
        self.assertEqual(reply["model"], "fake-reranker")
        self.assertEqual(reply["results"], [{"index": 1, "relevance_score": 1.0, "document": {"text": "basil basil basil"}},
                                            {"index": 2, "relevance_score": 0.3333, "document": {"text": "basil"}}])
        self.assertEqual(self.reranker.asked[0][2], "Find basil")
        bare = self.http.post("/v1/rerank", json={"query": "basil", "documents": ["a"], "return_documents": False})
        self.assertEqual(bare.json()["results"], [{"index": 0, "relevance_score": 0.0}])

    def test_bad_requests(self):
        for body in ({"query": "", "documents": ["a"]}, {"query": "q", "documents": []},
                     {"query": "q", "documents": [{"title": "a"}]}, {"query": "q", "documents": ["a"], "top_n": 0},
                     {"query": "q", "documents": ["a"] * 101}):
            self.assertEqual(self.http.post("/v1/rerank", json=body).status_code, 400, body)

    def test_no_reranker(self):
        http = TestClient(create_app(Engines({}, {}, {}), None, "none"), client=("127.0.0.1", 50000))
        self.assertEqual(http.post("/v1/rerank", json={"query": "q", "documents": ["a"]}).status_code, 500)


if __name__ == "__main__":
    unittest.main()
