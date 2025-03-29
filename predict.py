import argparse
from model.model import predict_dicom

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the class of a DICOM file.")
    parser.add_argument("image_path", type=str, help="Path to the DICOM file.")
    args = parser.parse_args()

    print("Predicting class for:", args.image_path)
    predicted_class = predict_dicom(args.image_path)
    print("Predicted Class:", predicted_class)
