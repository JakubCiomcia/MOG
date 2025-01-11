import numpy as np
import cv2
from MixtureOfGaussiansOptimized import MixtureOfGaussians, rgb2gray

K = 3  # liczba Gaussów
alpha = 0.5  # współczynnik uczenia
T = 0.85  # próg wag tła

# Open the input video file
cap = cv2.VideoCapture('rickroll.mkv')

# Get the frame dimensions and check if the file is loaded correctly
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Set scale factor (e.g., 0.5 for 50% reduction)
scale_factor = 0.33
width = 640
height = 360

print(f"Original Dimensions: {original_width}x{original_height}, FPS: {fps}")
print(f"Scaled Dimensions: {width}x{height}")

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' for better compatibility on Linux
out = cv2.VideoWriter('rickroll-opencv.avi', fourcc, fps, (width * 2, height), isColor=False)

mog = MixtureOfGaussians(height, width, K=K)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Scale down the frame
    scaled_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    frame_gray = rgb2gray(scaled_frame)
    mymask = mog.update(frame_gray, alpha, T)

    # Apply the background subtractor
    fgmask = fgbg.apply(scaled_frame)

    # Write the processed frame to the output file
    combined = np.hstack((fgmask, mymask))
    cv2.imshow('tut', combined)
    out.write(combined)

    # Optional: Display the scaled frame (uncomment if needed)
    # cv2.imshow('Foreground Mask', fgmask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
