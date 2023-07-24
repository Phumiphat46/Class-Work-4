import cv2 as cv
import numpy as np

# Load the images
input_img = cv.imread("4th/my_bass.png", cv.IMREAD_GRAYSCALE)
sobel_img = cv.imread("4th/sobel_x_magnitude_spectrum.png", cv.IMREAD_GRAYSCALE)

# Check if the images were loaded successfully
if input_img is None:
    raise ValueError("Failed to load 'my_bass.png'. Please check the file path.")
if sobel_img is None:
    raise ValueError("Failed to load 'sobel_x_magnitude_spectrum.png'. Please check the file path.")

# Check if both images have the same dimensions (optional but recommended)
if input_img.shape != sobel_img.shape:
    raise ValueError("The images must have the same dimensions.")

# Perform the dot product
dot_product_result = np.dot(input_img, sobel_img)

# Display or save the result as desired
cv.imshow("Dot Product Result", dot_product_result)
cv.waitKey(0)
cv.destroyAllWindows()

