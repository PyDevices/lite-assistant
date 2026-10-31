# Edit and rank photos

`POST /v1/images/edits` takes a photo as OpenAI's does, in an `image` field, or as `/v1/detect` takes one, and `model` says what to do to it:

- **`remove_background`** (ormbg) makes everything but the photo's subject transparent, as a PNG with an alpha channel, in under a second for a 640x480 photo.
- **`erase`** (MI-GAN) erases what `mask` covers, white where to erase, or what `prompt` names, which CLIPSeg finds: `prompt=the sunglasses`. What a mask's outline encloses is erased too, such as a pair of glasses' lenses. It took 0.7 s on a 512x512 photo, finding included.
- **`upscale`** (Real-ESRGAN) makes a photo 4 times as wide and tall, sharpening rather than blurring: 0.6 s for 160x120, 5.8 s for 640x480. It takes photos up to a million pixels.
- **`brighten`** (CPGA-Net) brightens a dark photo for a person to look at, at its full size, in about 50 ms. It's for people: it lightens a photo that isn't dark, and the detector does better with `/v1/detect`'s own `brighten`.

It returns the photo as OpenAI does, base64 in `data[0].b64_json`, or with `response_format=png` the PNG itself:

```bash
curl -F image=@dark.jpg -F model=brighten -F response_format=png http://localhost:5000/v1/images/edits -o bright.png
curl -F image=@beach.jpg -F model=erase -F "prompt=the sunglasses" -F response_format=png http://localhost:5000/v1/images/edits -o erased.png
```

In the chat, the [`edit_photo`](../reference/built-in-tools.md) tool makes these edits. Responses gives its photos as `image_generation_call` output items, as OpenAI's image generation tool does, after the tool's `mcp_call`; the chat page shows them, and a Save link downloads each.

`POST /v1/images/quality` scores photos from 1 to 10 with NIMA: `technical` for sharpness, noise and exposure, and `aesthetic` for how pleasing it is. Several `file` fields score a burst, and `best` is the index of the technically best, the one a board would keep:

```bash
curl -F file=@1.jpg -F file=@2.jpg -F file=@3.jpg http://localhost:5000/v1/images/quality
```

```json
{"object": "list", "model": "nima-mobilenet", "best": 0, "data": [
  {"index": 0, "technical": 3.82, "aesthetic": 4.44},
  {"index": 1, "technical": 3.02, "aesthetic": 3.61},
  {"index": 2, "technical": 3.79, "aesthetic": 3.68}]}
```

The scores are best compared with each other: the same scene blurred scored 3.02 against the sharp photo's 3.82.
