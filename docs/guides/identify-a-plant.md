# Identify a plant

`POST /v1/images/classifications` names what a photo shows from an image classification engine's list, taking the photo as `/v1/detect` does. PlantNet's list is 1081 species of plant; `model=plant` picks it by what it names, and `top` (1 to 20, default 5) says how many to give:

```bash
curl -F file=@flowers.jpg -F top=3 http://localhost:5000/v1/images/classifications
```

```json
{"object": "list", "model": "plantnet-300k-resnet18", "data": [
  {"label": "Calendula officinalis L.", "score": 0.655},
  {"label": "Freesia refracta (Jacq.) Klatt", "score": 0.151},
  {"label": "Tagetes erecta L.", "score": 0.1}]}
```

It sees the center of the photo, so fill the frame with one plant, a leaf or a flower. It names a plant for any photo, even one without a plant: a dog in grass was 80% a bee orchid.
