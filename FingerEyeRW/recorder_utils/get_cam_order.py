import av
import cv2
import numpy as np
import time

dev_addr = "/dev/video10" # 12 for index; 0 for ring; 2 for middle 
# === Open camera with PyAV (FFmpeg) ===
# - format=v4l2: use Video4Linux2
# - input_format=mjpeg: force MJPEG mode
# - video_size: set resolution
# - framerate: request 60 fps
container = av.open(
    dev_addr,
    format="v4l2",
    options={
        "input_format": "mjpeg",
        "video_size": "1280x480",
        "framerate": "25"
    }
)

print("Camera opened via PyAV")

# === FPS measurement ===
frame_count = 0
start_time = time.time()

for frame in container.decode(video=0):
    # Convert AV frame → numpy BGR (for OpenCV display)
    img = frame.to_ndarray(format="bgr24")

    # Show window
    cv2.imshow("PyAV Camera", img)

    # FPS counter
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

container.close()
cv2.destroyAllWindows()