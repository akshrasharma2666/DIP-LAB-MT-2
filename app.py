import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
st.set_option('deprecation.showfileUploaderEncoding', False)
from keras.models import load_model

html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Digital Image Processing lab</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
       Transformation and Reflection using GUI 
         """
         )
file= st.file_uploader("Please upload image", type=("jpg", "png"))


import cv2
from  PIL import Image, ImageOps
def import_and_predict(image_data):
     
     #direction = st.selectbox('direction', ('along x-axis', 'along y-axis'))
     #st.write('You selected: ', direction)
     if direction == "along y-axis":

         img_RE = cv2.imread("/content/img1.png")
  # convert from BGR to RGB so we can plot using matplotlib
         image3 = cv2.cvtColor(img_RE, cv2.COLOR_BGR2RGB)
  # disable x & y axis
  #plt.axis('off')
  # show the image
         #plt.imshow(image3)
         #plt.show()
  # get the image shape
         rows, cols, dim = image3.shape
  #  transformation matrix for x-axis reflection 
         M = np.float32([[-1,  0, cols],
                        [0, 1, 0],
                        [0,  0, 1   ]])

  # apply a perspective transformation to the image
         reflected_img = cv2.warpPerspective(image3,M,(int(cols),int(rows)))
  # disable x & y axis
  #plt.axis('off')
  # show the resulting image
         plt.imshow(reflected_img)
         plt.show()
         st.image(reflected_img, caption='reflected image', use_column_width=True)
  # save the resulting image to disk
  #plt.imsave("city_reflected.jpg", reflected_img)

     elif direction =="along x-axis":

        img_RE = cv2.imread("/content/img1.png")
  # convert from BGR to RGB so we can plot using matplotlib
        image3 = cv2.cvtColor(img_RE, cv2.COLOR_BGR2RGB)
  # disable x & y axis
  #plt.axis('off')
  # show the image
        plt.imshow(image3)
        plt.show()
  # get the image shape
        rows, cols, dim = image3.shape
  # transformation matrix for x-axis reflection 
        M = np.float32([[1,  0, 0],
                        [0, -1, rows],
                        [0,  0, 1   ]])

  # apply a perspective transformation to the image
        reflected_img = cv2.warpPerspective(image3,M,(int(cols),int(rows)))
  # disable x & y axis
  #plt.axis('off')
  # show the resulting image
        plt.imshow(reflected_img)
        plt.show()
        st.image(reflected_img, caption='reflected image', use_column_width=True)
  # save the resulting image to disk
  # plt.imsave("city_reflected.jpg", reflected_img)
   
     return 0

if file is None:
  st.text("Please upload an Image file")
else:
  file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)
  st.image(file,caption='Uploaded Image.', use_column_width=True)
    
if st.button("Display image"):
  direction = st.selectbox('direction', ('along x-axis', 'along y-axis'))
  st.write('You selected: ', direction)
  result=import_and_predict(image)
  
if st.button("About"):
  st.header(" LALITA SHARMA ")
  st.subheader("Student, Department of Computer Engineering")
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Digital Image processing Experiment</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)