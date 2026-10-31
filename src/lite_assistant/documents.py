"""
The text of documents: PDFs, Word files and photos of pages, for search_notes
to search alongside plain notes, and for files attached to a chat message.
"""
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from typing import List, Optional, Tuple

# The kinds of file read as documents, besides plain text
DOCUMENTS = (".pdf", ".docx")
PHOTOS = (".png", ".jpg", ".jpeg", ".webp")
# The most pages of a PDF read
MAX_PAGES = 300
# A PDF page with fewer characters of text than this is taken to be a scan, and read by OCR
SCANNED = 20
# The resolution a scanned page is rendered at for OCR, in dots per inch
SCAN_DPI = 150
# The most characters of an attached file's text the model is given
MAX_ATTACHED = 60000


def photo_text(image: bytes, ocr) -> str:
    """
    The text in a photo of a page, with a blank line between paragraphs,
    where the gap between two rows is more than half a row high.
    """
    from .engines import as_text
    lines = ocr.read(image)
    paragraphs: List[List] = []
    for line in lines:
        last = paragraphs[-1][-1] if paragraphs else None
        if last and line.box[1] - last.box[3] <= (last.box[3] - last.box[1]) / 2:
            paragraphs[-1].append(line)
        else:
            paragraphs.append([line])
    return "\n\n".join(as_text(paragraph) for paragraph in paragraphs)


def scanned_page(path: str, page: int, ocr) -> str:
    """ A PDF page's text read by OCR, if pdftoppm (from poppler) can render it. """
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm or not ocr:
        return ""
    with tempfile.TemporaryDirectory() as folder:
        made = subprocess.run([pdftoppm, "-f", str(page), "-l", str(page), "-r", str(SCAN_DPI), "-png",
                               "-singlefile", path, os.path.join(folder, "page")], capture_output=True, timeout=60)
        rendered = os.path.join(folder, "page.png")
        if made.returncode or not os.path.exists(rendered):
            return ""
        with open(rendered, "rb") as f:
            return photo_text(f.read(), ocr)


def pdf_pages(path: str, ocr=None) -> List[str]:
    """ Each page's text, scanned pages read by OCR when there's an OCR engine. """
    from pypdf import PdfReader
    pages = []
    for number, page in enumerate(PdfReader(path).pages[:MAX_PAGES], 1):
        text = (page.extract_text() or "").strip()
        if len(text) < SCANNED:
            text = scanned_page(path, number, ocr) or text
        # pypdf ends lines where the page does; a paragraph is where a line ends a sentence or a blank one
        pages.append(re.sub(r"(?<![.:!?])\n(?!\n)", " ", text))
    return pages


def docx_text(path: str) -> str:
    """ A Word file's paragraphs, each on its own, from its XML. """
    from xml.etree import ElementTree
    word = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        body = ElementTree.fromstring(archive.read("word/document.xml"))
    return "\n\n".join("".join(t.text or "" for t in p.iter(f"{word}t")) for p in body.iter(f"{word}p"))


def document_text(path: str, ocr=None) -> List[Tuple[str, str]]:
    """
    A file's text, as (where, text) parts: each page of a PDF ("page 3"),
    or the whole of anything else (""). A photo needs an OCR engine.
    """
    kind = os.path.splitext(path)[1].lower()
    if kind == ".pdf":
        return [(f"page {n}", text) for n, text in enumerate(pdf_pages(path, ocr), 1)]
    if kind == ".docx":
        return [("", docx_text(path))]
    if kind in PHOTOS:
        if not ocr:
            return []
        with open(path, "rb") as f:
            return [("", photo_text(f.read(), ocr))]
    with open(path, encoding="utf-8", errors="replace") as f:
        return [("", f.read())]


def attached_text(filename: str, data: bytes, ocr=None) -> str:
    """
    A file attached to a chat message, as text for the model: a PDF, Word
    file or photo of a page read as search_notes reads them, or any other
    file that is text. Raises ValueError for one with no text to give.
    """
    kind = os.path.splitext(filename)[1].lower()
    if kind in DOCUMENTS or kind in PHOTOS:
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "attached" + kind)
            with open(path, "wb") as f:
                f.write(data)
            try:
                parts = document_text(path, ocr)
            except Exception as e:
                raise ValueError(f"{filename} couldn't be read: {e}")
    elif b"\0" in data[:8192]:
        raise ValueError(f"{filename} isn't a document or a text file")
    else:
        parts = [("", data.decode("utf-8", errors="replace"))]
    text = "\n\n".join(f"[{where}]\n{part}" if where else part for where, part in parts if part.strip())
    if not text.strip():
        needs = " Reading photos of pages needs an OCR model." if kind in PHOTOS and not ocr else ""
        raise ValueError(f"No text was found in {filename}.{needs}")
    if len(text) > MAX_ATTACHED:
        text = text[:MAX_ATTACHED] + f"\n\n[The rest of {filename}, {len(text) - MAX_ATTACHED:,} more characters, was left out.]"
    return f"The attached file {filename}:\n\n{text}"
