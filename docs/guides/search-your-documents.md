# Search your documents

`search_notes` reads PDFs, page by page, so a result says where it is, such as `dishwasher_manual.pdf, page 3`; Word files; and photos of pages (`.png`, `.jpg`, `.webp`), with PP-OCRv6, a blank line between paragraphs where the rows leave a gap. A PDF page with no text, such as a scan, is read the same way, if poppler's `pdftoppm` is installed. Each file is read and embedded the first time a search sees it, and again when it changes, so the first search after adding a long PDF takes a while.

EmbeddingGemma finds the 8 paragraphs closest in meaning, and Qwen3-Reranker reads each with the question and puts them in order, about 3 seconds more. Asked how long a dishwasher's warranty was, which page said, and what a scanned table said about a garden bed, the chat found each one.

`POST /v1/rerank` does the reranking alone, for a board or client with its own documents, up to 100 at a time. `top_n` keeps the best, `return_documents: false` leaves out their text, and beyond Cohere's API, `instruction` says what relevant means, if not answering the query:

```bash
curl http://localhost:5000/v1/rerank -H "Content-Type: application/json" -d '{"query": "How long is the warranty?", "top_n": 2,
  "documents": ["Clean the dishwasher filter every month.", "The warranty covers parts and labour for two years.", "Basil wilts quickly."]}'
```

```json
{"object": "list", "model": "qwen3-reranker-0.6b",
 "results": [{"index": 1, "relevance_score": 0.9874, "document": {"text": "The warranty covers parts and labour for two years."}},
             {"index": 0, "relevance_score": 0.0001, "document": {"text": "Clean the dishwasher filter every month."}}]}
```

IBM's granite-docling, which turns a page into structured text, was tried and not added: on every page tried, it got at most the title and first sentence right, then repeated a line or made up text.
