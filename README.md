# ComfyUI Model Downloader

A Python-based model downloader for ComfyUI workflows with support for resume, parallel downloads, and automatic organization.

## Features

- ✅ **Resume interrupted downloads** - Pick up where you left off
- ✅ **Skip existing files** - Automatically detect and skip already downloaded models
- ✅ **Progress bars** - Real-time download progress with file sizes
- ✅ **Parallel downloads** - Download multiple models simultaneously
- ✅ **File size verification** - Ensure downloads completed successfully
- ✅ **Automatic folder structure** - Models are organized into correct ComfyUI directories
- ✅ **Workflow-based configuration** - Define model lists per workflow in JSON
- ✅ **GPU-specific configs** - Reference configurations for different VRAM capacities

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/comfyui-models.git
cd comfyui-models

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Download all models for WAN 2.2 workflow
python download_models.py workflows/wan22.json

# Download only required models
python download_models.py workflows/wan22.json --required-only

# Specify ComfyUI installation path
python download_models.py workflows/wan22.json --comfyui-path /path/to/ComfyUI

# Use sequential downloads (slower but more stable)
python download_models.py workflows/wan22.json --workers 1

# Skip file size verification
python download_models.py workflows/wan22.json --no-verify
```

## Usage

### Basic Command

```bash
python download_models.py <workflow_json> [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--comfyui-path PATH` | Path to ComfyUI installation | `/app/ComfyUI` |
| `--required-only` | Only download required models | `False` |
| `--workers N` | Number of parallel downloads | `3` |
| `--no-verify` | Skip file size verification | `False` |

### Examples

**Download models for Qwen Image Edit workflow:**
```bash
python download_models.py workflows/qwen_image_edit.json
```

**Download to custom ComfyUI location:**
```bash
python download_models.py workflows/wan22.json --comfyui-path ~/ComfyUI
```

**Download 5 models in parallel:**
```bash
python download_models.py workflows/wan22.json --workers 5
```

## Directory Structure

```
comfyui-models/
├── download_models.py          # Main download script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── workflows/                  # Workflow-specific model lists
│   ├── wan22.json             # WAN 2.2 workflow models
│   └── qwen_image_edit.json   # Qwen image editing models
└── configs/                    # GPU-specific configurations
    ├── gpu_24gb.json          # 24GB VRAM config (RTX 4090, 3090)
    └── gpu_96gb.json          # 96GB VRAM config (RTX Pro 6000)
```

## Creating Custom Workflow Configs

Create a new JSON file in the `workflows/` directory:

```json
{
  "workflow_name": "My Custom Workflow",
  "description": "Description of what this workflow does",
  "models": {
    "checkpoints": [
      {
        "name": "model_name.safetensors",
        "url": "https://huggingface.co/user/repo/resolve/main/model.safetensors",
        "size_gb": 6.5,
        "required": true,
        "destination": "models/checkpoints/"
      }
    ],
    "loras": [
      {
        "name": "lora_name.safetensors",
        "url": "https://civitai.com/api/download/models/12345",
        "size_gb": 0.2,
        "required": false,
        "destination": "models/loras/"
      }
    ]
  }
}
```

### Model Types

ComfyUI supports these model categories (use as keys in the `models` object):

- `checkpoints` - Main model files (SD, SDXL, Flux, etc.)
- `loras` - LoRA models
- `vae` - VAE models
- `clip` - CLIP models
- `text_encoders` - Text encoder models (T5, CLIP-L, etc.)
- `controlnet` - ControlNet models
- `upscale_models` - Upscaler models (ESRGAN, etc.)
- `embeddings` - Textual inversion embeddings
- `diffusion_models` - Diffusion model components

### Model Properties

| Property | Type | Description | Required |
|----------|------|-------------|----------|
| `name` | string | Filename to save as | ✅ |
| `url` | string | Download URL | ✅ |
| `size_gb` | float | Expected file size in GB | ❌ |
| `required` | boolean | Whether model is required | ❌ (default: true) |
| `destination` | string | Relative path from ComfyUI root | ❌ (default: models/{type}/) |

## GPU Configuration Reference

Use the configs in `configs/` as reference for your GPU setup:

### 24GB VRAM (RTX 4090, RTX 3090)
```bash
# Recommended settings
- FP16 precision
- Batch size: 1-2
- Can handle most SDXL workflows
```

### 96GB VRAM (RTX Pro 6000 Ada, H100)
```bash
# Recommended settings
- FP32 precision available
- Large batch sizes
- Multiple models can stay loaded
- No memory optimizations needed
```

## Finding Model URLs

### HuggingFace
```
https://huggingface.co/{user}/{repo}/resolve/main/{filename}
```

### CivitAI
```
https://civitai.com/api/download/models/{model_id}
```

### GitHub Releases
```
https://github.com/{user}/{repo}/releases/download/{tag}/{filename}
```

## Docker Usage

If using with the [ComfyUI Blackwell Docker image](https://github.com/laoda414/comfyui-blackwell):

```bash
# Run inside the container
docker exec -it comfyui-container bash
cd /app
git clone https://github.com/yourusername/comfyui-models.git
cd comfyui-models
pip install -r requirements.txt
python download_models.py workflows/wan22.json
```

Or mount the downloader as a volume:

```bash
docker run -v $(pwd)/comfyui-models:/models comfyui-blackwell
```

## Troubleshooting

### Download fails with "Connection timeout"
```bash
# Reduce parallel downloads
python download_models.py workflows/wan22.json --workers 1
```

### File size mismatch errors
```bash
# Skip verification if the model works fine
python download_models.py workflows/wan22.json --no-verify
```

### Out of disk space
```bash
# Download only required models first
python download_models.py workflows/wan22.json --required-only
```

### Models not appearing in ComfyUI
- Verify `--comfyui-path` points to correct installation
- Check that destination paths match ComfyUI's expected structure
- Restart ComfyUI to refresh model list

## Contributing

Contributions are welcome! To add a new workflow configuration:

1. Create a JSON file in `workflows/`
2. Follow the structure shown in existing examples
3. Test the download with your config
4. Submit a pull request

## License

MIT License - Feel free to use and modify for your needs.

## Related Projects

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The main ComfyUI repository
- [ComfyUI Blackwell Docker](https://github.com/laoda414/comfyui-blackwell) - Docker image for Blackwell GPUs

## Support

- For model downloader issues: [Open an issue](https://github.com/yourusername/comfyui-models/issues)
- For ComfyUI issues: [ComfyUI Issues](https://github.com/comfyanonymous/ComfyUI/issues)
