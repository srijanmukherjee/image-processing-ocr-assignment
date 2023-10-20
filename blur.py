from cv2.typing import MatLike
from typing import Tuple
import cv2 as cv

def average_blurring(src: MatLike, kernelSize: Tuple[int, int] = (3, 3)) -> MatLike:
    """returns blurred image using average blurring technique"""
    blurred = cv.blur(src, kernelSize)
    return blurred    

def gaussian_blurring(src: MatLike, kernelSize: Tuple[int, int] = (3, 3), sigmaX: float = 0) -> MatLike:
    """returns blurred image using gaussian blurring technique"""
    blurred = cv.GaussianBlur(src, kernelSize, sigmaX)
    return blurred

def median_blurring(src: MatLike, kernelSize: int) -> MatLike:
    """returns blurred image using median blurring technique"""
    blurred = cv.medianBlur(src, kernelSize)
    return blurred

def bilateral_blurring(src: MatLike, diameter: int = 11, sigmaColor: float = 21, sigmaSpace: float = 7):
    """returns blurred image using bilateral blurring"""
    blurred = cv.bilateralFilter(src, diameter, sigmaColor, sigmaSpace)
    return blurred

BLUR_OPTIONS = [
    'average',
    'gaussian',
    'median',
    'bilateral'
]