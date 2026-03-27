from pynq import Overlay, MMIO
import time
import cv2
import numpy as np
import urllib.request
import ssl
from IPython.display import display
from PIL import Image

# ==========================================
# 1. Helper Functions & AI Model Training
# ==========================================
def deskew(img):
    """Straightens out slanted handwriting using image moments."""
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * 20 * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (20, 20), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def get_hog():
    """Sets up the Histogram of Oriented Gradients feature extractor."""
    winSize = (20,20)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True
    return cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)

print("Downloading dataset and training SVM Model...")
url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/digits.png"
context = ssl._create_unverified_context()
resp = urllib.request.urlopen(url, context=context)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
digits_img = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

cells = [np.hsplit(row, 100) for row in np.vsplit(digits_img, 50)]
train_cells = [list(map(deskew, row)) for row in cells]

hog = get_hog()
hog_descriptors = []
for row in train_cells:
    for cell in row:
        hog_descriptors.append(hog.compute(cell))

train_data = np.squeeze(hog_descriptors).astype(np.float32)
responses = np.repeat(np.arange(10), 500)[:, np.newaxis].astype(np.int32)

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(train_data, cv2.ml.ROW_SAMPLE, responses)
print("Model ready!")

# ==========================================
# 2. Load the Hardware Design & Mappings
# ==========================================
print("Loading Hardware Overlay...")
overlay = Overlay("/home/xilinx/pynq/overlays/gesture/design_1_wrapper.bit")
arith = MMIO(0x43C00000, 0x1000)
switches_gpio = overlay.axi_gpio_0

STREAM_URL = "http://10.0.241.25:8080/video"
cap = cv2.VideoCapture(STREAM_URL)

# ==========================================
# 3. Reusable Capture and Predict Function
# ==========================================
def capture_and_predict(cap_device, svm_model, hog_extractor):
    # Flush the camera buffer to ensure we get a fresh frame
    for _ in range(5):
        cap_device.read()

    ret, frame = cap_device.read()
    if not ret:
        print("Frame capture failed")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 50]

    if len(valid_contours) == 0:
        print("No valid digit detected. Please try again.")
        return None

    cnt = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    margin = 5
    y1, y2 = max(0, y-margin), min(thresh.shape[0], y+h+margin)
    x1, x2 = max(0, x-margin), min(thresh.shape[1], x+w+margin)
    digit_roi = thresh[y1:y2, x1:x2]

    size = max(digit_roi.shape[0], digit_roi.shape[1])
    square = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - digit_roi.shape[0]) // 2
    x_offset = (size - digit_roi.shape[1]) // 2
    square[y_offset:y_offset+digit_roi.shape[0], x_offset:x_offset+digit_roi.shape[1]] = digit_roi

    digit_resized = cv2.resize(square, (20,20), interpolation=cv2.INTER_AREA)
    digit_deskewed = deskew(digit_resized)

    # Show what was captured
    display(Image.fromarray(digit_deskewed))

    hog_desc = hog_extractor.compute(digit_deskewed)
    sample = np.float32(hog_desc).reshape(-1, len(hog_desc))

    _, result = svm_model.predict(sample)
    return int(result[0][0])

# ==========================================
# 4. Hardware Execution Flow
# ==========================================

print("\nSetup ready. Waiting for Button 0 (BTN0) to be pressed...")

# Poll Button 0 while actively emptying the camera buffer
while switches_gpio.channel2[0].read() == 0:
    cap.grab() # Throws away stale frames so the stream stays live

num1 = capture_and_predict(cap, svm, hog)
print(f"--> First number captured: {num1}\n")

# Brief pause to let you move your hand, while keeping the buffer clear
timeout = time.time() + 1.0
while time.time() < timeout:
    cap.grab()

print("Waiting for Button 1 (BTN1) to be pressed...")

# Poll Button 1 while actively emptying the camera buffer
while switches_gpio.channel2[1].read() == 0:
    cap.grab() # Throws away stale frames so the stream stays live

num2 = capture_and_predict(cap, svm, hog)
print(f"--> Second number captured: {num2}\n")

# Only proceed if both numbers were successfully read
if num1 is not None and num2 is not None:
    print("Writing to FPGA Programmable Logic...")

    # Write to AXI Lite Registers via MMIO
    arith.write(0x00, num1)
    arith.write(0x04, num2)

    # Read the result back from the PL
    pl_result = arith.read(0x08)

    print("\n========================")
    print(f"Hardware Addition Result: {num1} + {num2} = {pl_result}")
    print("========================\n")

cap.release()
