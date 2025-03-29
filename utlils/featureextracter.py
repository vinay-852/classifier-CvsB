import numpy as np
from skimage.feature import hog
import mahotas as mt

def extract_features(images):
    features = []
    for img in images:
        # HOG Features
        hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        # GLCM Features
        glcm = mt.features.haralick(img.astype(np.uint8))
        glcm_mean = glcm.mean(axis=0)
        # Combine Features
        combined_features = np.hstack((hog_features, glcm_mean))
        features.append(combined_features)
    return np.array(features)
