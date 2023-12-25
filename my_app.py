import streamlit as st
import cv2
from PIL import Image
import numpy as np


if __name__=="__main__":

    # img_path = "Z:\\Datasets\\bone_fracture_dataset\\test\\images\\4_jpg.rf.31657c59b73817566ddb372f37d0db09.jpg"
    # original_img = cv2.imread(img_path)
    # modified_img = original_img.copy()
    original_img = None

    st.title('Bone Fracture Detection')

    col1, col2, col3 = st.columns([1, 1, 1])

    analyze_clicked = False

    with col1:
        uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()

            # To convert to a PIL Image:
            original_img = Image.open(uploaded_file)
            modified_img = np.array(original_img.copy())
            
            # Display the image
            # st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button("analyze"):
            analyze_clicked = True

    with col2:
        if original_img:
            st.image(original_img)

    with col3:

        if analyze_clicked:
            cv2.circle(modified_img, [15, 15], 10, [0, 255, 0], 2)
            st.image(modified_img)
            analyze_clicked = False



    

