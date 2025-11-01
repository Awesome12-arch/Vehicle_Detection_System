from pathlib import Path
import cv2
import time

save_path = Path(r"C:\Projects\VehicleEntrySystem\captured_images\Honda Amaze_Front Side_White.heif")

# ensure directory exists
save_path.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
time.sleep(1)

if not cap.isOpened():
    print("❌ Camera not opened. Check camera index or permissions.")
else:
    # warm up camera frames
    ret = False
    frame = None
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            break

    if not ret or frame is None:
        print("❌ Failed to capture image!")
    else:
        written = cv2.imwrite(str(save_path), frame)
        if written:
            print(f"✅ Image captured and saved at: {save_path}")
        else:
            print("❌ cv2.imwrite failed (format or codec not supported). Try .jpg or .png")
cap.release()
cv2.destroyAllWindows()
