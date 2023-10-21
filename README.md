# Image processing for OCR

Basic image processing before using image for OCR

## Instructions
```bash
python -m venv env
source ./env/bin/activate
python main.py <image>
```

> **NOTE:** <br> 
> Minimum python version requirement: **3.10**

### Usage
```
usage: main.py [-h] [-v] [-b {average,gaussian,median,bilateral}] [-k KERNEL] image

positional arguments:
  image                 path to image

options:
  -h, --help            show this help message and exit
  -v, --verbose
  -b {average,gaussian,median,bilateral}, --blur {average,gaussian,median,bilateral}
                        Choose blurring techinque
  -k KERNEL, --kernel KERNEL
                        set kernel size (must be odd), the kernel will be set as (size x size)
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