
import streamlit as st
import tensorflow as tf
import numpy as np

class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",	"Sneaker",	"Bag",	"Ankle boot"]

st.set_option("deprecation.showfileUploaderEncoding",False)
@st.cache_resource()
def load_modal():
  model=tf.keras.models.load_model('model_fc.hdf5',compile=False)
  model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
  return model

model = load_modal()

st.write("""
        # Fashion classification
        """)

file = st.file_uploader("Please upload an fashion image",type=["jpg","png","jpeg"])

import cv2
from PIL import ImageOps,Image

def import_and_predict(image_data,model):
  size=(28,28)
  image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
  img=np.asarray(image)
  img_reshape = img[np.newaxis]
  # st.image(img)
  pred = model.predict(img_reshape)
  return pred

if file is None:
  st.text("Upload the image")
else:
  image=Image.open(file)
  st.image(image,use_column_width=True)
  image = image.convert("L")
  # new_image = image.resize((28,28))
  # st.image(image)
  pred = import_and_predict(image,model)
  string = class_labels[np.argmax(pred)]
  st.success(string)
