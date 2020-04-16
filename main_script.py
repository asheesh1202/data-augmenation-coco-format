#!/usr/bin/env python
# coding: utf-8

# In[33]:


import os
import sys
import pandas as pd
import numpy as np
import cv2
import argparse
import os
#from google.cloud import automl_v1beta1 as automl


# In[34]:



from datetime import datetime


# In[35]:


#setting up the path where I am storing my labeled videos, careful about  the slash (/)
project_path = "G:\\waste management\\inpid_files\\video_files\\"
data_images_Extracted = "G:\\waste management\\inpid_files\\images"
bucket_name = "edge_device"
#home_folder = os.path.expanduser('~')
#os.chdir(os.path.join(home_folder, project_path))
# home = os.environ['HOME'] , I think this is for linux.
#os.getcwd() --> This is to check the working directory


# In[36]:


#here we check the files in our path, we would read the video files and change them into images
labels=[]
video_extract_path=project_path+'/'+''
file = os.listdir(video_extract_path)
for _ in file:
    labels.append(_.split('.')[0])


# In[ ]:


#Now we extract the images from the video
# Read the video from specified path 
try: 
    # creating a folder 
    if not os.path.exists(data_images_Extracted): 
        os.makedirs(data_images_Extracted)

# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data')
      
data_array = []

for label in labels:
    subdirpath = data_images_Extracted + "/" + label
    try: 
        # creating a folder 
        if not os.path.exists(subdirpath): 
            os.makedirs(subdirpath)

    # if not created then raise error 
    except OSError: 
        print ('Error: Creating sub-directory of data')
       
    data_filename = label + ".mp4"
    
    print(data_filename)
    cam = cv2.VideoCapture( project_path + data_filename)
    currentframe = 0
    
    while(True): 
        ret,frame = cam.read() 
        if ret: 
            # if video is still left continue creating images 
            now = datetime.now()
            current_time = now.strftime("%H_%M_%S")
            object_name = data_images_Extracted+ "/" + label + "/" +label + "_frame" + str(currentframe)+"_" +current_time +'.jpg'

            cv2.imwrite(object_name, frame) 

            #upload to google storage - This will be done later due to high data requirement
#             upload(src_name = object_name, dst_name = object_name)
# The CSV file will be created in a different way.
#             gcs_path = 'gs://' + bucket_name + "/" + object_name 
#             data_array.append((gcs_path, label))

            # increasing counter so that it will 
            # show how many frames are created 
            currentframe += 1
        else:
            
            break

# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows()


# In[17]:


labels


# In[ ]:


#method 1 to create a CSV : now we will make a CSV file to have the GCP path and the label


# In[ ]:


dataframe=pd.DataFrame(data_array)
dataframe.to_csv('demo_data.csv')


# In[9]:





# In[ ]:





# In[ ]:




