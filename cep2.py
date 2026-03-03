import cv2
import pytesseract
import pyttsx3
import numpy as np
from PIL import Image

# Path to tesseract executable (⚠️ adjust if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\locally_stored_imp_files\projects\TesseractOCR\tesseract.exe"

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 125)
tts_engine.setProperty("volume", 1.0)

def capture_and_process_from_phone():
    # Replace with your phone camera IP (from DroidCam/IP Webcam app)
    phone_ip_stream = "http://192.168.1.69:4747/video"   # 🔹 change this to your IP

    cap = cv2.VideoCapture(phone_ip_stream)

    if not cap.isOpened():
        raise Exception("Could not open phone camera stream")

    print("📷 Phone Camera Connected (WiFi)")
    print("👉 Press SPACE to capture & read text")
    print("👉 Press ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Phone Camera (WiFi)", frame)
        key = cv2.waitKey(1)

        # ESC to quit
        if key == 27:
            print("❌ Exiting...")
            break

        # SPACE = Capture
        if key == 32:
            print("\n✅ Image captured from phone camera")

            # --- OCR Processing ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pil_image = Image.fromarray(thresh)

            text = pytesseract.image_to_string(pil_image, config="--psm 6").strip()

            # --- Output ---
            if text:
                print("\n📖 Extracted Text:\n", text)
                print("🔊 Speaking text...")
                tts_engine.say(text)
                tts_engine.runAndWait()
            else:
                print("⚠️ No text detected")
                tts_engine.say("No text detected")
                tts_engine.runAndWait()

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("🚀 Starting OCR + Text-to-Speech using Phone Camera via WiFi...\n")
    capture_and_process_from_phone()

if __name__ == "__main__":
    main()
