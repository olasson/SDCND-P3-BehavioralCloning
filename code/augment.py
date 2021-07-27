import numpy as np
import cv2

def adjust_brightness(yuv_image, brightness_factor):
    """
    Adjust brightness of a YUV image
    
    Inputs
    ----------
    yuv_image: numpy.ndarray
        Array containing a single YUV image
    brightness_factor: float
        Scalar value to darken (lower values) or brighten (higher values) 
       
    Outputs
    -------
    yuv_image_out: numpy.ndarray
        YUV image with adjusted brightness. 
        
    """
    
    # Select Y-channel
    Y = yuv_image[:, :, 0]

    Y = np.where(Y * brightness_factor <= 255, Y * brightness_factor, 255)

    yuv_image_out = np.copy(yuv_image)

    yuv_image_out[:, :, 0] = Y

    return yuv_image_out

def translate_image(image, T_x):
    """
    Translate image scene
    
    Inputs
    ----------
    image: numpy.ndarray
        Array containing a single RGB image
    T_x: int
        Number of pixels ranslation in the x-dir (along a row)
       
    Outputs
    -------
    image: numpy.ndarray
        Translated image, dimensions preserved
        
    """

    border_pad = abs(T_x)

    image = cv2.copyMakeBorder(image, 0, 0, border_pad, border_pad, cv2.BORDER_REPLICATE)

    n_rows, n_cols, _ = image.shape

    T_matrix = np.float32([[1, 0, T_x],
                            [0, 1, 0]])

    image = cv2.warpAffine(image, T_matrix, (n_cols, n_rows))
    
    return image[:, border_pad:n_cols - border_pad]