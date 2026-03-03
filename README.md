# OCR Text-to-Speech System

This project captures text from images and converts it into speech using Python. It has two implementations for image acquisition: one using the laptop webcam and another using a smartphone camera via IP streaming (DroidCam). Both versions use **OpenCV** for image capture, **Tesseract OCR** for text recognition, and **pyttsx3** for text-to-speech output.

---

## Features

- Capture images from webcam or smartphone camera
- Preprocess images for optimal OCR:
  - Grayscale conversion
  - Gaussian blurring
  - Otsu’s thresholding
- Extract text from images using Tesseract OCR (`--psm 6`)
- Convert extracted text into speech using `pyttsx3`
- Cross-platform audio playback

---

## Versions

### 1. `cep.py` (Webcam Version)

- Uses the laptop’s built-in webcam (`cv2.VideoCapture(0)`) to capture frames
- Real-time capture triggered by pressing **SPACE**
- Text extraction and TTS pipeline remains standard
- Limitations:
  - Image quality depends on webcam sensor
  - Blurred frames can reduce OCR accuracy

---

### 2. `cep2.py` (Phone Camera Version)

- Replaces the webcam with a smartphone camera using DroidCam/IP Webcam
- Connects via IP stream (`http://<phone_ip>:4747/video`)
- Captures higher-quality frames for improved OCR accuracy
- All other processing steps (preprocessing, OCR, TTS) remain the same
- Advantages:
  - Better focus and resolution
  - Wireless operation via WiFi
  - More reliable text recognition

---

## Installation

```bash
pip install opencv-python pytesseract pyttsx3 pillow numpy
