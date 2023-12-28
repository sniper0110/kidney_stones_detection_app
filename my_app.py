import streamlit as st
import cv2
from PIL import Image
import numpy as np
from detection_model import KidneyStonesDetectionModel




if __name__=="__main__":

    model_path = "./ks_detection.pt"
    model = KidneyStonesDetectionModel(model_path=model_path)

    original_img = None

    st.title('Kidney Stones Detection')

    col1, col2, col3 = st.columns([1, 1, 1])

    analyze_clicked = False

    with col1:
        uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()

            # To convert to a PIL Image:
            original_img = Image.open(uploaded_file)
            modified_img = np.array(original_img.copy()) # This is an RGB image
            
            # Display the image
            # st.image(image, caption='Uploaded Image.', use_column_width=True)

            if st.button("analyze"):
                analyze_clicked = True

    with col2:
        if original_img:
            st.image(original_img)

    with col3:

        if analyze_clicked:
            
            print("Running inference..")
            model.run_inference(image=original_img)
            # model.run_inference(image=modified_img)

            print("Drawing results on image..")
            image_with_detections = model.draw_bboxes_on_image(image=modified_img)

            st.image(image_with_detections)
            analyze_clicked = False



    

