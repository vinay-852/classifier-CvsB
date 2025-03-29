import cv2

IMG_SIZE = 128  # Resize images

def preprocess_image(img):
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Gaussian Blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Median Filtering
    img = cv2.medianBlur(img, 3)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    # Normalize
    img = img / 255.0
    return img
