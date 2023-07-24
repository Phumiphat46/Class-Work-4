import cv2 as cv
import numpy as np

# Load the image in grayscale (provide the full file path if needed)
image = cv.imread('1st/my_bass.png', cv.IMREAD_GRAYSCALE)

# Check if the image is loaded properly
if image is None:
    print("Error: Image not found or could not be loaded.")
else:
    # Create Sobel filter kernels
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2],
                        [-1, 0, 1]])

    # Apply the Sobel filter in the horizontal direction
    sobel_x_output = cv.filter2D(image, -1, sobel_x)

    # Save the filtered image
    cv.imwrite('sobel_x_output.png', sobel_x_output)