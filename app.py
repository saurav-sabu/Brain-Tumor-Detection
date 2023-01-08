import streamlit as st
import tensorflow as tf
from tensorflow import keras
import cv2

st.title("Brain Tumor Classification (MRI)")

model2 = keras.models.load_model("model_final.h5")
model1 = keras.models.load_model("model.h5")

st.sidebar.title("Choose the Model:")
option = st.sidebar.selectbox("Select model",["model1","model2"])

if option == "model2":
    st.header("Model 2")
    file = st.file_uploader("Upload the MRI Image")
    if file is not None:
        print(file)
        save_image_path = "./upload_images/"+file.name
        with open(save_image_path,"wb") as f:
            f.write(file.getbuffer())
        img = cv2.imread(save_image_path)
        img = cv2.resize(img,(150,150))
        img_input = img.reshape((1,150,150,3))
        y_pred = model2.predict(img_input).argmax(axis=1)
        
        if y_pred[0] == 0:
            st.write("Target: glioma_tumor")
        elif y_pred[0] == 1:
            st.write("Target: meningioma_tumor")
        elif y_pred[0] == 2:
            st.write("Target: no_tumor")
        else:
            st.write("Target: pituitary_tumor")

        st.subheader("Accuracy: 0.9892")
        st.subheader("Validation Accuracy: 0.8850")
    else:
        st.warning("Not Entered image")

if option == "model1":
    st.header("Model 1")
    file = st.file_uploader("Upload the MRI Image")
    if file is not None:
        print(file)
        save_image_path = "./upload_images/"+file.name
        with open(save_image_path,"wb") as f:
            f.write(file.getbuffer())
        img = cv2.imread(save_image_path)
        img = cv2.resize(img,(150,150))
        img_input = img.reshape((1,150,150,3))
        y_pred = model1.predict(img_input).argmax(axis=1)
        
        if y_pred[0] == 0:
            st.write("Target: glioma_tumor")
        elif y_pred[0] == 1:
            st.write("Target: meningioma_tumor")
        elif y_pred[0] == 2:
            st.write("Target: no_tumor")
        else:
            st.write("Target: pituitary_tumor")

        st.subheader("Accuracy: 0.9274")
        st.subheader("Validation Accuracy: 0.8367")
    else:
        st.warning("Not Entered image")
