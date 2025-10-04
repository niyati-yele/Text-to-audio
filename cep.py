import cv2
import pytesseract
import pyttsx3
import numpy as np
from PIL import Image
import tempfile
import os

#local path to the tesseract executable(my device's path)
pytesseract.pytesseract.tesseract_cmd = r'"C:\locally_stored_imp_files\projects\TesseractOCR'

# Initializing tts engine
tts_engine = pyttsx3.init()
#rate of the speech and volume of the speech
tts_engine.setProperty('rate', 125)
tts_engine.setProperty('volume', 1.0)

def capture_image_from_camera():
    #Capture a single frame from the webcam
    cap = cv2.VideoCapture(0)  #cap is a VideoCapture object used to read from the camera.

    if not cap.isOpened(): #check if the cameera is opened successfully
        raise Exception("Could not open webcam")

    print("Press SPACE to capture image, ESC to exit...")

    while True:
        ret, frame = cap.read() #ret is boolean indicating if the frame was read correctly and frame is the frame captured
        if not ret:
            continue

        cv2.imshow("Live Camera - Press SPACE to Capture", frame) #shows the video capturing frame in the window
        key = cv2.waitKey(1) #wait for the key to be pressed for 1 milisecond and return the ASCII value of the key pressed

        if key == 27:  # ESC key to exit (ASCII value of ESC is 27)
            print("Exiting...")
            cap.release() #release the camera resource
            cv2.destroyAllWindows() #closes the openCv windows
            exit() #exits the program

        if key == 32:  # SPACE to capture(ASCII value of space is 32)
            print("IMAGE CAPTURED ^_^")
            cap.release() #release the camera resource
            cv2.destroyAllWindows() #closes the openCv windows
            return frame #return the captured frame

def process_image_for_ocr(frame):
    #Preprocess image and extract text
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert the image to grayscale

    # Blur and threshold
    gray = cv2.GaussianBlur(gray, (3, 3), 0) #apply guassian blur to the image
    #(this convert image to black&white image and shades of gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #apply thresholding to get a binary image

    # Convert to PIL Image for Tesseract
    pil_image = Image.fromarray(thresh) # convert the numpy array to PIL image

    # OCR
    text = pytesseract.image_to_string(pil_image, config='--psm 6') #perform OCR on the image using tesseract
    # psm 6 assumes a single uniform block of text
    return text.strip() #strip removes any leading/tailing whitespace characters

def speak_text(text):
    #Convert text to speech and play

    if not text: # if empty text
        print("No text to convert to speech.")
        return #exit

    print("\nConverting to speech...") 
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio: #create a temporary file to save the audio
        audio_path = temp_audio.name #path of the temporary audio file

    tts_engine.save_to_file(text, audio_path) #save speech to temporary file
    tts_engine.runAndWait() #process & play speech

    # Play audio using system default player (cross-platform method can vary)
    print("Playing audio...")
    if os.name == 'nt':  # windows
        os.system(f'start {audio_path}') #opens the audio file with the deafult audio player
    elif os.name == 'posix':  # Linux, macOS
        os.system(f'xdg-open "{audio_path}"')  # On macOS you may need: os.system(f'afplay "{audio_path}"')

def main():
    print("Starting real-time OCR and TTS...\n") #start the program
    frame = capture_image_from_camera() #capture image from camera

    print("\nProcessing image for text recognition...") #process the image for OCR
    text = process_image_for_ocr(frame) #extaract text from the iamge

    print("\nExtracted Text :- ") #display the extracted text 
    if text: #text not empty
        print(text)
    else:
        print("No text detected in the image.")

    speak_text(text) #convert the text to speech and play it 

if __name__ == "__main__":
    main()
