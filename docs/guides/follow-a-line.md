# Follow a line

`POST /v1/lines` finds the line to follow in a photo from a camera looking ahead, taking it as `/v1/detect` does. By default it's tape or paint on a floor, darker, lighter or a different color than the floor; `model=lanes` finds a road's lane markings with Ultra-Fast-Lane-Detection instead, and the path is the middle of the car's own lane.

```bash
curl --data-binary @floor.jpg -H "Content-Type: image/jpeg" http://localhost:5000/v1/lines
```

```json
{"object": "lines", "model": "line-finder", "found": true,
 "path": {"offset": -0.257, "angle": 15.3, "ahead": 0.137, "points": [[0.3717, 0.975], [0.3806, 0.9292], ...]},
 "lines": [[[0.3717, 0.975], [0.3806, 0.9292], ...]]}
```

- **`offset`** is where the line is at the bottom of the photo, nearest the car: -1 at the left edge, 0 in the middle and 1 at the right. Steer toward it.
- **`angle`** is how many degrees the line leans right of straight up the photo, over its nearest half. A straight line off to one side leans toward the middle too, from perspective.
- **`ahead`** is the offset of the farthest point found, to see a curve coming.
- **`points`** are the path as fractions of the photo from the top left, nearest first; `lines` are every line found, left to right. The line finder looks at the bottom 70% of the photo, where the floor is.
- **`found`** is false, and `path` null, when there's no line, or for lanes, no lane marking on one side.
- **`response_format=jpeg`** returns the photo with the lines drawn on it in yellow and the path in green.

The line finder was tried on drawn floors: black, white and blue tape, straight and curving, on planks whose seams it ignores. It takes the line that stands out most, so a floor with other strong edges, such as a rug's, may confuse it. Ultra-Fast-Lane-Detection doesn't see tape on a floor at all, and it found the lanes in a dashcam photo of a highway.
