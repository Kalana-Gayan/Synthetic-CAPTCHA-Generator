
# Synthetic CAPTCHA Generator for AI/ML Datasets

A powerful, highly-configurable Python script for generating synthetic CAPTCHA images. This tool is designed to create robust, large-scale datasets for training machine learning and computer vision models.

This script can be used to reverse-engineer and replicate complex visual styles, including multi-layer noise, advanced text distortion, and procedural artifacts.

**Author:** Kalana Gayan Senevirathne

## Key Features

* **Full Dataset Generation:** Automatically generate thousands of images with `train/val/test` splits and a corresponding `labels.csv` file, ready for any ML pipeline.
* **Highly Configurable:** Nearly every visual parameter is exposed in the script's settings block, allowing for precise control over fonts, colors, noise, and distortions.
* **Advanced Text Distortion:**
    * Word-level sine wave distortion.
    * Per-character pixel warping, rotation, and shearing.
    * Precise control over character spacing and positioning.
* **Multi-Layer Noise Engine:**
    * Procedural 2x2 "cluster dot" noise with configurable density and color palettes.
    * Semi-transparent, randomly angled line noise.
* **Built with a Pro-Stack:** Uses a fast and efficient stack of **OpenCV**, **Pillow (PIL)**, and **NumPy** for all image manipulations.

## Example Output

The generator can be configured to produce a wide variety of styles. Here are a few examples created by the script:

| Generated Example 1 | Generated Example 2 |
| :---: | :---: |
| ![Example 1](000091_BONK.png) | *(Add another image here, e.g., `examples/example2.png`)* |

*(This is a great place to add a grid of 4-6 of your best generated images)*

## Installation

This project requires Python 3 and the following libraries.

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  Install the required libraries:
    ```bash
    pip install opencv-python pillow numpy tqdm
    ```

3.  (Optional) Create a `requirements.txt` file with the following:
    ```
    opencv-python
    Pillow
    numpy
    tqdm
    ```
    And install with: `pip install -r requirements.txt`

## Usage

The script is run from the command line and has two primary modes.

### 1. Generating a Full Dataset (Recommended)

To generate a complete dataset (e.g., 1000 images), use the `--total` argument.

```bash
python captcha_generator.py --total 1000
```
This will:

    - Create a captcha_dataset directory.

    - Automatically split the 1000 images into train, val, and test sub-folders.

    - Generate a labels.csv file in the root, mapping each image file to its correct label.

The labels used for generation can be edited in the DEFAULT_LABELS_LIST inside the script.

### 2. Generating a Single Test Image

To quickly test a single word, use the --single argument.
```Bash

python captcha_generator.py --single "FARTCOIN"
```
This will create a single image named test_FARTCOIN.png in the project's root directory.

### Configuration

All visual parameters can be changed by editing the ⚙️ FINAL SETTINGS block at the top of captcha_generator.py.

This includes:

    - Canvas size

    - Font files (.ttf) and sizes

    - Text colors

    - All distortion, wave, and shear amplitudes

    - Line noise properties (count, color, width)

    - Cluster dot noise properties (density, colors)

    - Final blur and rotation effects

Tweak these variables to create entirely new CAPTCHA styles.

###License

This project is open-source. Please feel free to use, modify, and distribute. (Consider adding an MIT License file to your repository).
