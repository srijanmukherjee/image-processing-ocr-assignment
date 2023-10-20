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
    
    print(f"{height=}")
    print(f"{width=}")
    print(f"{channels=}")
    print(f"{kernel=}")

    # STEP 1: apply blurring/smoothening
    # if blur_technique is not None:
    #     print(f"applying {blur_technique} blurring")
    #     match blur_technique:
    #         case 'average':
    #             img = blur.average_blurring(img, kernel)
    #         case 'gaussian':
    #             img = blur.gaussian_blurring(img, kernel)
    #         case 'median':
    #             img = blur.median_blurring(img, kernel_size)
    #         case 'bilateral':
    #             img = blur.bilateral_blurring(img)

    # STEP 2: convert to binary image
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    threshold, img = cv.threshold(img, 150, 255, cv.THRESH_BINARY)

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
    threshold, img = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
    cv.imshow("image", img)
    # cv.waitKey(0)
    while cv.waitKeyEx() != 27:
        continue

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
