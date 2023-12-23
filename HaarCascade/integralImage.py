import numpy as np
"""
Integral image : each pixel is the sum of all pixels in the original image 
that are before (left and above) the pixel.

Original    Integral
+--------   +------------
| 1 2 3 .   | 0  0  0  0 .
| 4 5 6 .   | 0  1  3  6 .
| . . . .   | 0  5 12 21 .
            | . . . . . .

"""
def calc_integral_image (img):
    """
    Calculate the integral image of a 2D numpy array.

    Parameters:
    - img (numpy.ndarray): 2D numpy array of size (h, w).

    Returns:
    numpy.ndarray: Integral image of size (h+1, w+1).
    """
    
    h,w = img.shape
    integral_image = np.zeros((h+1,w+1))
    for y in range(1,h+1):
        for x in range(1,w+1):
            integral_image[y, x] = integral_image[y, x-1] + integral_image[y-1, x] - integral_image[y-1, x-1] + img[y-1, x-1]
    return integral_image

def calc_region_sum (integral_image, x0, y0, w, h):
    """
    Calculate the sum of pixel values within a specified rectangle in the integral image.

    Parameters:
    - integral_image (numpy.ndarray): 2D numpy array of size (l+1, w+1).
    - x0 (int): Starting column of the rectangle.
    - y0 (int): Starting row of the rectangle.
    - w (int): Width of the rectangle.
    - h (int): Height of the rectangle.

    Returns:
    int: Sum of all pixels in the specified rectangle.
    
    Raises:
    ValueError: If the specified rectangle extends beyond the bounds of the integral_image.
    """
    if x0 < 0 or y0 < 0 or x0 + w >= integral_image.shape[1] or y0 + h >= integral_image.shape[0]:
        raise ValueError("Rectangle is not entirely within the bounds of the integral_image")
    
    return integral_image[int(y0+h), int(x0+w)] - integral_image[int(y0+h), int(x0)] - integral_image[int(y0), int(x0+w)] + integral_image[int(y0), int(x0)]