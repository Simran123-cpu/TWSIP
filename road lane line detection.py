import cv2
import numpy as np

# Open a video file or use 0 for webcam
cap = cv2.VideoCapture('C:/Users/DELL/Downloads/lane vid.mp4')  # Replace 'video.mp4' with your video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define vertices of the polygon for the ROI
    roi_vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)

    # Create a mask for the ROI
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)

    # Apply the mask to the edges
    masked_edges = cv2.bitwise_and(edges, mask)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    # Draw detected lines on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Display the output frame
    cv2.imshow('Lane Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
