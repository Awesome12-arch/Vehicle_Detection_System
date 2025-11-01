import cv2
import easyocr
from pathlib import Path
import matplotlib.pyplot as plt

# ✅ Set your input and output paths
image_path = r"C:\Projects\VehicleEntrySystem\captured_images\Honda_Amaze_Front_White.jpg"
output_path = r"C:\Projects\VehicleEntrySystem\captured_images\cropped_plate.jpg"

# Load image
image = cv2.imread(image_path)
if image is None:
    print("❌ Error: Could not load image. Check file path.")
    exit()

# Load pre-trained Haar Cascade for Indian license plates
cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(cascade_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect plates
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 25))

if len(plates) == 0:
    print("⚠️ No plate detected. Try a clearer front or rear image.")
else:
    print(f"✅ Detected {len(plates)} possible plate(s).")

    for (x, y, w, h) in plates:
        plate_roi = image[y:y + h, x:x + w]
        cv2.imwrite(output_path, plate_roi)
        print(f"📸 Cropped plate saved at: {output_path}")

        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show detected plate
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Plate Region")
    plt.axis("off")
    plt.show()

    # OCR on cropped plate
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(output_path)

    print("\n🧾 OCR Results:")
    if not results:
        print("⚠️ No text detected in cropped plate.")
    else:
        for (_, text, prob) in results:
            print(f"Detected Plate Number: {text} (Confidence: {prob:.2f})")
