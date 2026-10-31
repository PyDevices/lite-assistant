# Find things by name

`POST /v1/segment` finds things by name with CLIPSeg, in a photo taken as `/v1/detect` takes one. `prompt` names one thing, or several separated by commas (in JSON, a list). Each comes back `found` or not, and if found, with its `box` and `center` as fractions of the photo, the `area` of the photo it covers, the `score` at its surest point, and its `distance` in meters from MoGe-2 (`distances=false` skips that):

```bash
curl -F file=@dog.jpg -F "prompt=the dog, the grass, a car" http://localhost:5000/v1/segment
```

```json
{"object": "segmentation", "model": "clipseg-rd64-refined", "depth_model": "moge-2-vits", "units": "meters", "data": [
  {"prompt": "the dog", "found": true, "score": 0.964, "box": [0.111, 0.213, 0.841, 0.767], "center": [0.344, 0.535], "area": 0.1918, "distance": 3.18},
  {"prompt": "the grass", "found": true, "score": 0.794, "box": [0.0, 0.182, 1.0, 0.997], "center": [0.568, 0.654], "area": 0.5467, "distance": 2.79},
  {"prompt": "a car", "found": false}]}
```

- **`threshold`** (0 to 1, default 0.4) is how sure CLIPSeg must be that a point is the thing.
- **`response_format=png`** returns the first thing's mask instead, at the photo's size, white where it is.

Unlike `/v1/detect`, it finds anything that has a name, but one region a name, not each of several objects, and it knows what things are, not where: "the grass behind the dog" finds all the grass. The `estimate_distance` tool uses it for things YOLOX doesn't know.
