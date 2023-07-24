import cv2 as cv
import numpy as np

# Load the image
image = cv.imread('my_bass.jpg', cv.IMREAD_GRAYSCALE)

# Apply the Sobel filter
sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)

# Apply the Fourier Transform
sobel_x_freq = np.fft.fft2(sobel_x)
sobel_y_freq = np.fft.fft2(sobel_y)

# Shift the zero-frequency component to the center
sobel_x_freq_shifted = np.fft.fftshift(sobel_x_freq)
sobel_y_freq_shifted = np.fft.fftshift(sobel_y_freq)

# Calculate the magnitude spectrum
magnitude_spectrum_x = np.abs(sobel_x_freq_shifted)
magnitude_spectrum_y = np.abs(sobel_y_freq_shifted)

# Apply logarithmic transformation
magnitude_spectrum_x = np.log1p(magnitude_spectrum_x)
magnitude_spectrum_y = np.log1p(magnitude_spectrum_y)

# Normalize the magnitude spectra for visualization
magnitude_spectrum_x = cv.normalize(magnitude_spectrum_x, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
magnitude_spectrum_y = cv.normalize(magnitude_spectrum_y, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

# Save the magnitude spectra as images
cv.imwrite('sobel_x_magnitude_spectrum.png', magnitude_spectrum_x)
cv.imwrite('sobel_y_magnitude_spectrum.png', magnitude_spectrum_y)

