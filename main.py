from cv2.typing import MatLike
from skimage.morphology import skeletonize
from skimage.util import invert
from skimage import img_as_ubyte
import matplotlib.pyplot as plt 
import numpy as np
import cv2 as cv
import argparse
import sys

import blur

"""
TODO:
- Save the results into png file with appropriate file prefix
    ex. input file: E1.png
        output file: <output directory>/E1_blurred_<technique_name>.png (for Q1)
                     <output directory>/E1_binary_64x64.png             (for Q2)
                     <output directory>/E1_skeletonized_64x64.png       (for Q3)
- Automatically compute the binary image threshold (https://stackoverflow.com/a/20075082)
- Save the final result (skeletonized binary 64x64 image) in xlsx format (for the assigment)
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
    args = vars(ap.parse_args())

    if args.get('verbose') == False:
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

    # STEP 2: convert to binary image
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshold, img = cv.threshold(img, binary_threshold, 255, cv.THRESH_BINARY)

    # STEP 3: resize to 64x64
    resized_img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)
    # cv.imshow("resized image", resized_img)

    # STEP 4: skeletonize
    img = invert(img)
    skeleton = skeletonize(img)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)
    
    ax = axes.ravel()

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('original', fontsize=20)

    ax[1].imshow(skeleton, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('skeleton', fontsize=20)

    fig.tight_layout()
    plt.show()

    img = cv.resize(img_as_ubyte(invert(skeleton)), (64, 64), interpolation=cv.INTER_AREA)
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshold, img = cv.threshold(img, 250, 255, cv.THRESH_BINARY)
    cv.imshow("image", img)
    # cv.waitKey(0)
    while cv.waitKeyEx() != 27:
        continue

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
