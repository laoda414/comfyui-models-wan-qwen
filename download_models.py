#!/usr/bin/env python3
"""
ComfyUI Model Downloader
Downloads models for ComfyUI workflows based on JSON configurations.

Features:
- Resume interrupted downloads
- Skip existing files
- Progress bars with tqdm
- Parallel downloads
- File size verification
- Automatic folder structure creation
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import requests
from tqdm import tqdm


class ModelDownloader:
    def __init__(self, comfyui_path: str, max_workers: int = 3, verify_size: bool = True):
        """
        Initialize the model downloader.

        Args:
            comfyui_path: Path to ComfyUI installation
            max_workers: Number of parallel downloads
            verify_size: Whether to verify file sizes after download
        """
        self.comfyui_path = Path(comfyui_path)
        self.max_workers = max_workers
        self.verify_size = verify_size

        if not self.comfyui_path.exists():
            raise ValueError(f"ComfyUI path does not exist: {comfyui_path}")

    def load_workflow_config(self, config_path: str) -> Dict:
        """Load workflow configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)

    def get_file_size(self, url: str) -> Optional[int]:
        """Get remote file size in bytes."""
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            if 'Content-Length' in response.headers:
                return int(response.headers['Content-Length'])
        except Exception as e:
            print(f"Warning: Could not get file size for {url}: {e}")
        return None

    def download_file(self, url: str, destination: Path, expected_size_gb: Optional[float] = None) -> bool:
        """
        Download a file with progress bar and resume capability.

        Args:
            url: URL to download from
            destination: Local file path to save to
            expected_size_gb: Expected file size in GB (for verification)

        Returns:
            True if download was successful, False otherwise
        """
        # Create destination directory if it doesn't exist
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Check if file already exists and is complete
        if destination.exists():
            if expected_size_gb:
                actual_size_gb = destination.stat().st_size / (1024**3)
                if abs(actual_size_gb - expected_size_gb) < 0.1:  # Within 100MB tolerance
                    print(f"✓ {destination.name} already exists and size matches (skipping)")
                    return True
                else:
                    print(f"! {destination.name} exists but size mismatch (re-downloading)")
            else:
                print(f"✓ {destination.name} already exists (skipping)")
                return True

        # Get remote file size
        remote_size = self.get_file_size(url)

        # Check for partial download
        resume_header = {}
        initial_pos = 0
        if destination.exists():
            initial_pos = destination.stat().st_size
            resume_header = {'Range': f'bytes={initial_pos}-'}
            print(f"→ Resuming download from {initial_pos / (1024**2):.1f} MB")

        try:
            # Stream download with progress bar
            response = requests.get(url, headers=resume_header, stream=True, timeout=30)
            response.raise_for_status()

            total_size = remote_size or int(response.headers.get('Content-Length', 0))

            mode = 'ab' if initial_pos > 0 else 'wb'
            with open(destination, mode) as f:
                with tqdm(
                    total=total_size,
                    initial=initial_pos,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=destination.name
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Verify file size if expected size is provided
            if self.verify_size and expected_size_gb:
                actual_size_gb = destination.stat().st_size / (1024**3)
                if abs(actual_size_gb - expected_size_gb) > 0.1:
                    print(f"✗ Size mismatch: expected {expected_size_gb:.2f} GB, got {actual_size_gb:.2f} GB")
                    return False

            print(f"✓ Downloaded {destination.name}")
            return True

        except Exception as e:
            print(f"✗ Failed to download {destination.name}: {e}")
            return False

    def download_model(self, model: Dict, model_type: str) -> bool:
        """Download a single model."""
        url = model['url']
        filename = model['name']
        destination_dir = model.get('destination', f'models/{model_type}/')
        expected_size = model.get('size_gb')

        # Construct full destination path
        destination = self.comfyui_path / destination_dir / filename

        print(f"\n[{model_type}] {filename}")
        print(f"URL: {url}")
        print(f"Destination: {destination}")

        return self.download_file(url, destination, expected_size)

    def download_workflow_models(self, config_path: str, required_only: bool = False) -> None:
        """
        Download all models for a workflow.

        Args:
            config_path: Path to workflow JSON configuration
            required_only: Only download models marked as required
        """
        config = self.load_workflow_config(config_path)

        print(f"\n{'='*60}")
        print(f"Workflow: {config['workflow_name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}\n")

        # Collect all models to download
        download_tasks = []
        for model_type, models in config['models'].items():
            for model in models:
                if required_only and not model.get('required', True):
                    continue
                download_tasks.append((model, model_type))

        print(f"Total models to download: {len(download_tasks)}\n")

        # Download models
        if self.max_workers == 1:
            # Sequential download
            for model, model_type in download_tasks:
                self.download_model(model, model_type)
        else:
            # Parallel download
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.download_model, model, model_type): (model, model_type)
                    for model, model_type in download_tasks
                }

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        model, model_type = futures[future]
                        print(f"✗ Error downloading {model['name']}: {e}")

        print(f"\n{'='*60}")
        print("Download complete!")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Download ComfyUI models from workflow configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'workflow',
        help='Path to workflow JSON file (e.g., workflows/wan22.json)'
    )

    parser.add_argument(
        '--comfyui-path',
        default='/app/ComfyUI',
        help='Path to ComfyUI installation (default: /app/ComfyUI)'
    )

    parser.add_argument(
        '--required-only',
        action='store_true',
        help='Only download required models'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=3,
        help='Number of parallel downloads (default: 3, use 1 for sequential)'
    )

    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip file size verification'
    )

    args = parser.parse_args()

    try:
        downloader = ModelDownloader(
            comfyui_path=args.comfyui_path,
            max_workers=args.workers,
            verify_size=not args.no_verify
        )

        downloader.download_workflow_models(
            config_path=args.workflow,
            required_only=args.required_only
        )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
