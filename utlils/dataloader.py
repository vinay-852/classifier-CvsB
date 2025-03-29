import os
import pydicom
import numpy as np
from utlils.preprocess import preprocess_image

def load_dicom_images(main_dir, class_names):
    images, labels = [], []
    for label, class_name in enumerate(class_names):
        folder_path = os.path.join(main_dir, class_name)
        for file in os.listdir(folder_path):
            if file.endswith(".dcm"):
                dicom_path = os.path.join(folder_path, file)
                dicom_data = pydicom.dcmread(dicom_path)
                img = dicom_data.pixel_array
                img = preprocess_image(img)
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)
