import streamlit as st
import os
import pydicom
import matplotlib.pyplot as plt
from model.model import predict_dicom

# Directory containing test DICOM images
TEST_DIR = "data/Test"

def display_dicom_image(file_path):
    """Display a DICOM image."""
    dicom_data = pydicom.dcmread(file_path)
    image = dicom_data.pixel_array
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    st.pyplot(plt)

def main():
    st.title("DICOM Image Classifier")
    st.write("Select a DICOM file from the `data/Test` folder or upload a new one to predict its class.")

    # Option 1: Select a DICOM file from the test directory
    if os.path.exists(TEST_DIR):
        dicom_files = [f for f in os.listdir(TEST_DIR) if f.endswith(".dcm")]
        if dicom_files:
            selected_file = st.selectbox("Select a DICOM file from the test folder:", dicom_files)
            if selected_file:
                selected_file_path = os.path.join(TEST_DIR, selected_file)
                st.write("### Selected DICOM Image:")
                display_dicom_image(selected_file_path)
                if st.button("Predict Selected File"):
                    st.write("Predicting...")
                    predicted_class = predict_dicom(selected_file_path)
                    st.success(f"Predicted Class: {predicted_class}")
        else:
            st.write("No DICOM files found in the test folder.")

    st.write("---")

    # Option 2: Upload a custom DICOM file
    uploaded_file = st.file_uploader("Or upload a DICOM file", type=["dcm"])
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = os.path.join(TEST_DIR, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded image
        st.write("### Uploaded DICOM Image:")
        display_dicom_image(temp_file_path)

        # Predict the class
        st.write("Predicting...")
        predicted_class = predict_dicom(temp_file_path)
        st.success(f"Predicted Class: {predicted_class}")
        
        # Clean up the temporary file
        os.remove(temp_file_path)

if __name__ == "__main__":
    main()
