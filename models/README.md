# models

`realesr-general-x4v3.onnx`: Real-ESRGAN general x4 v3 (SRVGGNetCompact), used by `upscale.py`.

- Source: the official weights `realesr-general-x4v3.pth` from https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0 (SHA-256 8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292).
- Converted to ONNX (opset 17, dynamic height / width) with PyTorch; the output matches PyTorch's within 5e-6.
- Input: `image`, float32 RGB in 0-1, shape (1, 3, H, W). Output: `upscaled`, (1, 3, 4H, 4W).
- Licence: BSD-3-Clause, Copyright (c) 2021, Xintao Wang. https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE
