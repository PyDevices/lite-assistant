# Find objects and faces

`POST /v1/detect` takes a photo the same ways `/v1/depth` does, and returns the objects YOLOX finds in it, surest first, each with how far it is when a depth engine is loaded:

```bash
curl --data-binary @photo.jpg -H "Content-Type: image/jpeg" http://localhost:5000/v1/detect
```

For a photo of a living room, it returns (two of eleven objects shown):

```json
{"object": "detection", "model": "yolox-tiny", "depth_model": "moge-2-vits", "units": "meters",
 "objects": [
  {"label": "tv", "score": 0.906, "box": {"left": 0.0091, "top": 0.392, "right": 0.2437, "bottom": 0.6233},
   "center": {"x": 0.126, "y": 0.508}, "place": "left", "distance": 3.9},
  {"label": "vase", "score": 0.78, "box": {"left": 0.8584, "top": 0.7015, "right": 0.9151, "bottom": 0.9384},
   "center": {"x": 0.887, "y": 0.82}, "place": "bottom right", "distance": 1.64}]}
```

- **`box`** gives the object's edges, and `center` its middle, as fractions of the width and height from the top left. `place` says where that is in words, such as `left` or `bottom right`.
- **`distance`** is in meters: a low percentile of the middle of the box, so the background around the object or seen through it doesn't count. It's `null` where there's no estimate.
- **`model=yunet`** finds faces instead, each labeled `face`, with `points` for the eyes, nose and mouth corners: `right_eye`, `left_eye`, `nose`, `right_mouth` and `left_mouth`, as `x` and `y` fractions like the box's. Right and left are the person's own, so a face looking at the camera has its right eye on the photo's left. A face's distance tells a board how far away someone is. Each face also has its `expression` (angry, contemptuous, disgusted, afraid, happy, neutral, sad or surprised), with the three likeliest in `expressions`, from HSEmotion, and its `gaze` from L2CS-Net, degrees of `yaw` (negative toward the photo's right) and `pitch` from straight at the camera, with `looking_at_camera` true within 25 degrees both ways, so a board can wake when someone looks at it. `describe=false` skips them. Both misread faces under about 50 pixels tall.
- **`pose=true`** adds each person's `keypoints` from RTMPose: the 17 COCO joints, `nose`, `left_eye` and so on to `right_ankle`, each with `x`, `y` and a `score` from 0 to 1, and `raised_hands`, `left` or `right` (the person's own) for a wrist above its shoulder. About 10 ms a person.
- **`hands=true`** adds each person's `hands` too: for each hand RTMPose's hand model finds, found past the wrist along the forearm, its `side`, 21 `keypoints` and `fingers_up`, such as `["index", "middle"]`. Hands under 32 pixels are skipped, and fingers read best from a hand held up toward the camera.
- **`threshold`** (0 to 1) is how sure the model must be: by default 0.35 for objects and 0.6 for faces. Lower finds more, with more mistakes.
- **`distances=false`** skips the depth map, which takes most of the time: about 60 ms instead of half a second or more.
- **`brighten`** stretches a dark photo's contrast before the detector sees it: `auto`, the default, does it when the photo's mean brightness is under 20%, and `true` or `false` always or never. The reply says `"brightened": true` when it did. On COCO photos darkened to 7% with sensor noise, YOLOX found 41 of their 61 objects with it and 24 without.
