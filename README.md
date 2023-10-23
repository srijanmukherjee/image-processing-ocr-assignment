# Image processing for OCR

Basic image processing (smoothening, binarizing, skeletonizing) before using image for OCR

## Instructions
```bash
# start virtual environment
python -m venv env
source ./env/bin/activate

# install dependencies
pip install -r requirements.txt

# run the code
python main.py <image>

# Example
python main.py --blur gaussian -k 7 -v dataset/English/E3.png 

# Run over all examples in the dataset
python experiment.py
```

> **NOTE:** <br> 
> Minimum python version requirement: **3.10**

### Usage
```
usage: main.py [-h] [-v] [-b {average,gaussian,median,bilateral}] [-k KERNEL] [-o OUTPUT] image

positional arguments:
  image                 path to image

options:
  -h, --help            show this help message and exit
  -v, --verbose         Enable verbose output
  -b {average,gaussian,median,bilateral}, --blur {average,gaussian,median,bilateral}
                        Choose blurring techinque
  -k KERNEL, --kernel KERNEL
                        set kernel size (must be odd), the kernel will be set as (size x size)
  -o OUTPUT, --output OUTPUT
                        output directory
```

## What does it do?

- Apply image filtering technique to smoothen the image
    - Average blurring
    - Gaussian blurring
    - Median blurring
    - Bilateral blurring
- Convert into binary image
- Skeletonize the image
- Resize into 64x64 image

This example demonstrates some basic image processing techinques to be applied before we use the image for processing techniques like OCR.

## References
- https://pyimagesearch.com/2021/04/28/opencv-smoothing-and-blurring
- https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html