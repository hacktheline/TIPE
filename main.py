
'''
pip3 install opencv-python
pip3 install pytesseract
pip3 install imutils

pip3 install pyttsx3
sudo apt install espeak

'''


import cv2
import pytesseract
import time
import random
import string
import imutils
import pyttsx3


DEBUG = True
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract' # Mention the installed location of Tesseract-OCR in your system



print("""
------------------------------------------
               MV-HELPER
            TIPE 2021-2022
------------------------------------------
""")

def f__say(text):
    print("[*]Say: %s" %text)
    engine = pyttsx3.init() 
    engine.say("Hi, I am text to speach") 
    engine.setProperty('rate', 125) 
    engine.runAndWait()
    
def f__random():
    return str(int(time.time())) + '__' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

def f__ocrtext(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped = im2[y:y + h, x:x + w]
        text = pytesseract.image_to_string(cropped)
        if not text.replace("\n","").replace(" ","") == "":
            f__say("There is a writing %s " %text)
    return True

def f__person_body(image):
    haar_upper_body_cascade = cv2.CascadeClassifier("data/haarcascade_upperbody.xml")
    frame = imutils.resize(image, width=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    upper_body = haar_upper_body_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    if DEBUG:
        for (x, y, w, h) in upper_body:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) # creates green color rectangle with a thickness size of 1
            cv2.putText(frame, "Upper Body Detected", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # creates green color text with text size of 0.5 & thickness size of 2
        cv2.imwrite("image/ba.png", frame)
    if len(upper_body) != 0:
        f__say("there are %i people in front of you" % len(upper_body))
    return True

def f__person_face(image):
    haar_frontalface_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt.xml")
    frame = imutils.resize(image, width=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frontalface = haar_frontalface_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    if DEBUG:
        for (x, y, w, h) in frontalface:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) # creates green color rectangle with a thickness size of 1
            cv2.putText(frame, "Face Detected", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # creates green color text with text size of 0.5 & thickness size of 2
        cv2.imwrite("image/ba.png", frame)
    if len(frontalface) != 0:
        f__say("there are %i people looking at you" % len(frontalface))
        
def f__analyse(image):
    f__ocrtext(image)
    f__person_body(image)
    f__person_face(image)

def main():
    while True:
        cam = cv2.VideoCapture(0)
        name = f__random() + ".png"
        print(name)
        ret, image = cam.read()
        f__analyse(image)
        if DEBUG == True:
            cv2.imwrite("image/" + name, image)
        time.sleep(5)
  
main()
