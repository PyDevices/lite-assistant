# Read text in photos

`POST /v1/ocr` reads the text in a photo, taken as `/v1/detect` takes one, with PP-OCRv6. `text` is all of it, with words side by side in a row joined by spaces and rows by newlines; `lines` has each line the detector found, with how sure the reader is (`score`, 0 to 1) and its `box` as fractions of the photo (left, top, right, bottom):

```bash
curl --data-binary @sign.jpg -H "Content-Type: image/jpeg" http://localhost:5000/v1/ocr
```

```json
{"object": "ocr", "model": "pp-ocrv6-small", "text": "EXIT ONLY\nOpen 9am - 5pm Monday to Friday",
 "lines": [{"text": "EXIT ONLY", "score": 0.998, "box": [0.15, 0.21, 0.3, 0.26]},
           {"text": "Open 9am - 5pm Monday to Friday", "score": 0.996, "box": [0.15, 0.35, 0.59, 0.41]}]}
```

The detector sees the photo scaled to fit 640 pixels, and each line is read from the full photo, so text should be at least a fiftieth of the photo's height. Lines may be tilted or turned on their side. It reads print, not handwriting.
