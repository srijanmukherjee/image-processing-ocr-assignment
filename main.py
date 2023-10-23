import os
from pathlib import Path
from cv2.typing import MatLike
from skimage.morphology import skeletonize
from skimage.util import invert
from skimage import img_as_ubyte
import matplotlib.pyplot as plt 
import pandas as pd
import cv2 as cv
import argparse
import sys

import blur

"""
TODO:
- Automatically compute the binary image threshold (https://stackoverflow.com/a/20075082)
"""

def kernel_type(value: str):
    """validates that kernel size is odd"""
    value = int(value)
    if value <= 0:
        raise ValueError("must be positive")
    
    if value % 2 == 0:
        raise ValueError("must be an odd number")
    
    return value

def main():    
    # initialize arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('image', type=str, help='path to image')
    ap.add_argument('-v', '--verbose', default=False, action='store_true')
    ap.add_argument('-b', '--blur', choices=blur.BLUR_OPTIONS, help='Choose blurring techinque', default='median')
    ap.add_argument('-k', '--kernel', type=kernel_type, help='set kernel size (must be odd), the kernel will be set as (size x size)', default=3)
    ap.add_argument('-t', '--threshold', type=int, default=150, help='threshold for binarizing image')
    ap.add_argument('-o', '--output', type=str, default='result', help='output directory')
    args = vars(ap.parse_args())

    verbose = args.get('verbose')

    if not verbose:
        cv.setLogLevel(0)

    # load the image
    srcImage: MatLike | None = cv.imread(args.get('image'))
    if srcImage is None:
        print("failed to read image", file=sys.stderr)
        exit(1)

    height, width, channels = srcImage.shape[:3]

    img = srcImage.copy()
    kernel_size = args.get("kernel")
    blur_technique = args.get('blur')
    kernel = (kernel_size, kernel_size)
    binary_threshold = args.get('threshold')
    outdir = args.get('output')
    filename_without_ext = Path(args.get('image')).stem

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    if verbose:
        print(f"{height=}")
        print(f"{width=}")
        print(f"{channels=}")
        print(f"{kernel=}")

    # STEP 1: apply blurring/smoothening
    if blur_technique is not None:
        print(f"applying {blur_technique} blurring")
        match blur_technique:
            case 'average':
                img = blur.average_blurring(img, kernel)
            case 'gaussian':
                img = blur.gaussian_blurring(img, kernel)
            case 'median':
                img = blur.median_blurring(img, kernel_size)
            case 'bilateral':
                img = blur.bilateral_blurring(img)

        file_path = f'{outdir}/{filename_without_ext}_blurred_{blur_technique}.png'
        if verbose:
            print(f'Writing to {file_path}')
        cv.imwrite(file_path, img)

    # STEP 2: convert to binary image
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(img, binary_threshold, 255, cv.THRESH_BINARY)

    # STEP 3: resize to 64x64
    resized_img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)

    file_path = f'{outdir}/{filename_without_ext}_blurred_{blur_technique}_binary_64x64.png'
    if verbose:
        print(f'Writing to {file_path}')
    cv.imwrite(file_path, resized_img)

    # STEP 4: skeletonize
    img = invert(img)
    skeleton = skeletonize(img)

    # get 64x64 binary skeleton image
    img = cv.resize(img_as_ubyte(invert(skeleton)), (64, 64), interpolation=cv.INTER_AREA)
    _, img = cv.threshold(img, 250, 255, cv.THRESH_BINARY)

    file_path = f'{outdir}/{filename_without_ext}_blurred_{blur_technique}_skeleton_binary_64x64.png'
    if verbose:
        print(f'Writing to {file_path}')
    cv.imwrite(file_path, img)


    # Write skeletonize 64x64 img to spreadsheet
    df = pd.DataFrame(img)
    file_path = f'{outdir}/{filename_without_ext}.xlsx'
    df.to_excel(excel_writer=file_path)
    if verbose:
        print(f'Written to {file_path}')

if __name__ == '__main__':
    main()
