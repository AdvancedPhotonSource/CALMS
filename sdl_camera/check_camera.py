import cv2

print("Scanning for connected cameras...")
for i in range(10):  # Check indexes 0 to 4 to get the connected cameras
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        ret, frame = cap.read()
        if ret:
            print(f"Captured frame from camera {i}")
        else:
            print(f"Camera {i} opened but failed to capture")
        cap.release()
    else:
        print(f"No camera at index {i}")