import time
import urllib.request
from pathlib import Path

# Standard test images often used in image processing
TEST_IMAGES = {
    "coins.png": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.21.0/skimage/data/coins.png",  # Good grayscale test image
    "astronaut.png": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.21.0/skimage/data/astronaut.png",  # Good color test image
    "coffee.png": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.21.0/skimage/data/coffee.png",  # Good natural scene image
    "camera.png": "https://raw.githubusercontent.com/scikit-image/scikit-image/v0.21.0/skimage/data/camera.png",  # Classic test image
}


def download_test_images(max_retries=3, delay=1):
    """Download standard test images used in examples."""
    output_dir = Path(__file__).parent.parent / "examples" / "metrics" / "sample_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in TEST_IMAGES.items():
        output_path = output_dir / filename
        if not output_path.exists():
            print(f"Downloading {filename}...")
            success = False

            for attempt in range(max_retries):
                try:
                    urllib.request.urlretrieve(url, output_path)  # nosec B310
                    print(f"Saved to {output_path}")
                    success = True
                    break
                except urllib.error.HTTPError as e:
                    print(f"Attempt {attempt+1}/{max_retries} failed: HTTP Error {e.code}: {e.reason}")
                except urllib.error.URLError as e:
                    print(f"Attempt {attempt+1}/{max_retries} failed: URL Error: {e.reason}")

                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)

            if not success:
                print(f"Failed to download {filename} after {max_retries} attempts.")


if __name__ == "__main__":
    download_test_images()
