# Image-to-Image

Edit or transform input images with text prompts using vLLM-Omni's diffusion pipeline.

- `image_edit.py`: command-line script for image editing with advanced options.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Key Arguments](#key-arguments)
- [More CLI Examples](#more-cli-examples)
- [Advanced Features](#advanced-features)
- [FAQ](#faq)

## Overview

This folder provides an entrypoint for image editing with diffusion models using vLLM-Omni. Input one or more images along with a text prompt, and the model generates an edited image that follows the instruction.

### Supported Models

| Model | Image Shape | Peak VRAM (GiB) * | Model Weights (GiB) |
| ----- | ----------- | ----------------- | ------------------- |
| `Qwen/Qwen-Image-Edit` | 1024 × 1024 | 60.0 | 53.7 |
| `Qwen/Qwen-Image-Edit-2509` | 1024 × 1024 | 60.0 | 53.7 |
| `Qwen/Qwen-Image-Edit-2511` | 1024 × 1024 | 60.0 | 53.7 |
| `Qwen/Qwen-Image-Layered` | 1024 × 1024 | 60.0 | 53.7 |
| `meituan-longcat/LongCat-Image-Edit` | 1024 × 1024 | 71.2 | 27.3 |
| `OmniGen2/OmniGen2` | 1024 × 1024 | 20.1 | 14.7 |
| `black-forest-labs/FLUX.1-Kontext-dev` | 1024 × 1024 | 77.6 | 31.4 |

!!! info
    *Peak VRAM: based on basic single-card usage, batch size = 1, without any acceleration/optimization features. Some models may require CPU offloading on a single 80 GiB GPU.
    Model Weights: the VRAM consumption of model weights (BF16) is printed as `Model loading took xxx GiB and xxx seconds` in the logging.

Default model: `Qwen/Qwen-Image-Edit`

## Quick Start

Download the example image first:

```bash
wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png
```

### Python API

Single-image editing:

```python
from PIL import Image
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

if __name__ == "__main__":
    omni = Omni(model="Qwen/Qwen-Image-Edit")
    image = Image.open("qwen-bear.png").convert("RGB")
    outputs = omni.generate(
        {
            "prompt": "Let this mascot dance under the moon, surrounded by floating stars",
            "multi_modal_data": {"image": image},
        },
        OmniDiffusionSamplingParams(
            true_cfg_scale=4.0,
            num_inference_steps=50,
        ),
    )
    result = outputs[0].request_output.images[0]
    result.save("edited.png")
```

### Local CLI Usage

```bash
python image_edit.py \
  --image qwen-bear.png \
  --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
  --output output_image_edit.png \
  --num-inference-steps 50 \
  --cfg-scale 4.0
```

## Key Arguments

**Common arguments:**

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `--model` | str | `"Qwen/Qwen-Image-Edit"` | Model name or local path. Use `Qwen/Qwen-Image-Edit-2509` or later for multiple-image support |
| `--image` | str (one or more) | — | Path(s) to input image file(s) (PNG, JPG, etc.). Can specify multiple images |
| `--prompt` | str | — | Text prompt describing the edit to apply |
| `--negative-prompt` | str | `None` | Negative prompt for classifier-free guidance |
| `--seed` | int | `0` | Random seed for deterministic results |
| `--cfg-scale` | float | `4.0` | True CFG scale. Enabled when > 1 and `--negative-prompt` is set. Higher values follow the prompt more closely at the cost of quality |
| `--guidance-scale` | float | `1.0` | Guidance scale for guidance-distilled models. Enabled when > 1; ignored otherwise |
| `--num-inference-steps` | int | `50` | Number of denoising steps (more steps = higher quality, slower) |
| `--num-outputs-per-prompt` | int | `1` | Number of images to generate for the given prompt |
| `--output` | str | `"output_image_edit.png"` | Path to save the edited image. For `Qwen-Image-Layered`, used as a filename prefix; outputs are saved as `<prefix>_0.png`, `<prefix>_1.png`, etc. |
| `--resolution` | int | `640` | Bucket resolution in `{640, 1024}` that determines the condition and output resolution |
| `--vae-use-slicing` | flag | off | Enable VAE slicing for memory optimization |
| `--vae-use-tiling` | flag | off | Enable VAE tiling for memory optimization |
| `--cfg-parallel-size` | int | `1` | Set to `2` to enable CFG Parallel (requires 2 GPUs) |
| `--ulysses-degree` | int | `1` | Ulysses sequence parallel degree for multi-GPU inference |
| `--ring-degree` | int | `1` | Ring sequence parallel degree for multi-GPU inference |
| `--tensor-parallel-size` | int | `1` | Tensor parallel degree inside the DiT |
| `--enable-cpu-offload` | flag | off | Enable CPU offloading for diffusion models |
| `--enable-layerwise-offload` | flag | off | Enable layerwise (blockwise) offloading on DiT modules |
| `--enforce-eager` | flag | off | Disable `torch.compile` and force eager execution |
| `--cache-backend` | str | `None` | Cache acceleration backend: `"cache_dit"` or `"tea_cache"` |

> If you encounter OOM errors, try using `--vae-use-slicing` and `--vae-use-tiling` to reduce memory usage.

**OmniGen2-specific arguments:**

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `--guidance-scale-2` | float | `None` | Image guidance scale (`image_guidance_scale`). For OmniGen2, `--guidance-scale` acts as `text_guidance_scale` and `--guidance-scale-2` acts as `image_guidance_scale` |

**Qwen-Image-Layered-specific arguments:**

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `--layers` | int | `4` | Number of layers to decompose the output image into |
| `--color-format` | str | `"RGB"` | Color format for input/output images. Set to `"RGBA"` for layered output |

**Cache-DiT-specific arguments** (used when `--cache-backend cache_dit`):

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `--cache-dit-fn-compute-blocks` | int | `1` | Number of forward compute blocks. Optimized for single-transformer models |
| `--cache-dit-bn-compute-blocks` | int | `0` | Number of backward compute blocks |
| `--cache-dit-max-warmup-steps` | int | `4` | Maximum warmup steps (useful for few-step models) |
| `--cache-dit-residual-diff-threshold` | float | `0.24` | Residual diff threshold. Higher values enable more aggressive caching |
| `--cache-dit-max-continuous-cached-steps` | int | `3` | Maximum continuous cached steps to prevent precision degradation |
| `--cache-dit-enable-taylorseer` | flag | off | Enable TaylorSeer acceleration. Not suitable for few-step models |
| `--cache-dit-taylorseer-order` | int | `1` | TaylorSeer polynomial order |
| `--cache-dit-scm-steps-mask-policy` | str | `None` | SCM mask policy: `None` (disabled), `"slow"`, `"medium"`, `"fast"`, `"ultra"` |
| `--cache-dit-scm-steps-policy` | str | `"dynamic"` | SCM steps policy: `"dynamic"` or `"static"` |

**TeaCache-specific arguments** (used when `--cache-backend tea_cache`):

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `--tea-cache-rel-l1-thresh` | float | `0.2` | Threshold for accumulated relative L1 distance. Higher values cache more aggressively |

## More CLI Examples

### Multiple Image Editing

`Qwen/Qwen-Image-Edit-2509` and `Qwen/Qwen-Image-Edit-2511` support multiple input images. Pass multiple paths to `--image` to provide them:

```bash
python image_edit.py \
  --model Qwen/Qwen-Image-Edit-2511 \
  --image qwen-bear.png qwen-bear.png \
  --prompt "Place two bears side-by-side in a snowy mountain landscape" \
  --output output_combined.png \
  --num-inference-steps 50 \
  --cfg-scale 4.0 \
  --guidance-scale 1.0
```

### OmniGen2 Image Editing

OmniGen2 uses `guidance_scale` as `text_guidance_scale` and `guidance_scale_2` as `image_guidance_scale`:

```bash
python image_edit.py \
  --model OmniGen2/OmniGen2 \
  --image qwen-bear.png \
  --prompt "Change the background to a snowy forest at night." \
  --negative-prompt "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face" \
  --num-inference-steps 50 \
  --seed 0 \
  --guidance-scale 5.0 \
  --guidance-scale-2 2.0 \
  --output outputs/image_edit.png \
  --num-outputs-per-prompt 2
```

### Layered Editing (Qwen-Image-Layered)

`Qwen/Qwen-Image-Layered` decomposes the output into multiple RGBA layers:

```bash
python image_edit.py \
  --model Qwen/Qwen-Image-Layered \
  --image qwen-bear.png \
  --prompt "" \
  --output layered_output \
  --num-inference-steps 50 \
  --cfg-scale 4.0 \
  --layers 4 \
  --color-format RGBA
```

Output images are saved as `layered_output_0.png`, `layered_output_1.png`, etc.

### LongCat Image Editing

```bash
python image_edit.py \
  --model meituan-longcat/LongCat-Image-Edit \
  --image qwen-bear.png \
  --prompt "Add colorful fireworks bursting in the background behind the bear" \
  --output longcat_edit.png \
  --num-inference-steps 50 \
  --cfg-scale 4.0
```

## Advanced Features

### CFG Parallel

Set `--cfg-parallel-size 2` to enable CFG Parallel for faster inference on 2-GPU setups. This is especially effective for image editing workflows that rely on classifier-free guidance (`--cfg-scale > 1`).

```bash
python image_edit.py \
  --image qwen-bear.png \
  --prompt "Let this mascot dance under the moon" \
  --cfg-parallel-size 2 \
  --num-inference-steps 50 \
  --cfg-scale 4.0 \
  --output output_cfg_parallel.png
```

See more examples in the [diffusion parallelism user guide](../../../docs/user_guide/diffusion/parallelism_acceleration.md#cfg-parallel).

### Cache Acceleration

Use `--cache-backend` to speed up inference with minimal quality trade-off.

**Cache-DiT** (recommended for high quality):

```bash
python image_edit.py \
  --model Qwen/Qwen-Image-Edit-2511 \
  --image qwen-bear.png \
  --prompt "Add a white art board written with colorful text 'vLLM-Omni' on grassland" \
  --output output_cache_dit.png \
  --num-inference-steps 50 \
  --cfg-scale 4.0 \
  --cache-backend cache_dit \
  --cache-dit-max-continuous-cached-steps 3 \
  --cache-dit-residual-diff-threshold 0.24 \
  --cache-dit-enable-taylorseer
```

**TeaCache** (quick setup):

```bash
python image_edit.py \
  --image qwen-bear.png \
  --prompt "Let this mascot dance under the moon" \
  --cache-backend tea_cache \
  --tea-cache-rel-l1-thresh 0.25 \
  --output output_tea_cache.png
```

See [Cache-DiT guide](../../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md) and [TeaCache guide](../../../docs/user_guide/diffusion/cache_acceleration/teacache.md) for tuning details.

## FAQ

**Q: Which models support multiple input images?**

`Qwen/Qwen-Image-Edit-2509` and `Qwen/Qwen-Image-Edit-2511` support multiple images via `--image img1.png img2.png ...`. The base `Qwen/Qwen-Image-Edit` accepts only a single image.

**Q: What is the difference between `--cfg-scale` and `--guidance-scale`?**

`--cfg-scale` is the true classifier-free guidance (CFG) scale and requires a `--negative-prompt` to take effect. `--guidance-scale` is used by guidance-distilled models that accept guidance as a direct input parameter, and is independent of CFG.

**Q: I get OOM errors. What should I do?**

Try `--vae-use-slicing` and `--vae-use-tiling` first (no quality loss). If memory pressure persists, add `--enable-cpu-offload`. For layerwise offloading on a single card, use `--enable-layerwise-offload`.

**Q: How do I disable `torch.compile` for debugging?**

Add `--enforce-eager` to skip compilation and run in eager mode.
