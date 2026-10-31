# Measure distance in a photo

`POST /v1/depth` takes a photo, as the request body itself (`Content-Type: image/jpeg` or another image type), as a multipart `file`, or as JSON `{"image": "data:..."}` with a data: or http(s) URL. A board can send its camera's JPEG as it is:

```bash
curl --data-binary @photo.jpg -H "Content-Type: image/jpeg" "http://localhost:5000/v1/depth?columns=4&rows=3"
```

For a close photo of loaves of bread on a table, it returns:

```json
{"object": "depth", "model": "moge-2-vits", "units": "meters", "width": 448, "height": 299,
 "horizontal_fov": 64.5, "center": 0.71,
 "nearest": {"distance": 0.62, "x": 0.413, "y": 0.92}, "farthest": {"distance": 1.25, "x": 0.857, "y": 0.038},
 "grid": [[1.06, 1.0, 0.97, 1.12], [0.81, 0.71, 0.72, 0.84], [0.68, 0.62, 0.65, 0.74]], "coverage": 0.998}
```

- **Distances** are straight lines from the camera, in meters. `center` is the median over the middle tenth of the photo, like a rangefinder's reading.
- **`nearest` and `farthest`** skip the nearest and farthest 2% of points, so stray pixels don't count. `x` and `y` say where they are, as fractions of the width and height from the top left.
- **`grid`** is the median distance in each cell, row by row from the top; `columns` and `rows` (1 to 64, default 8) set its size. Options go in the query string, form fields or the JSON body.
- **`null`** means no estimate, such as sky. `coverage` is the share of the photo with one.
- **`horizontal_fov`** is the camera's field of view across the photo, in degrees, as the model judges it.
- **`response_format=png`** returns the depth map as a grayscale PNG instead: near is bright, far is dark, and no estimate is black. Shading follows inverse distance. The `X-Depth-Near` and `X-Depth-Far` headers give the distances of the brightest and darkest grays.

Distances from one photo are estimates. Expect them to be off by 10 to 20 percent in a room or a street. Close-ups with nothing around them for scale, such as a telephoto shot of a face, can be off by far more.
