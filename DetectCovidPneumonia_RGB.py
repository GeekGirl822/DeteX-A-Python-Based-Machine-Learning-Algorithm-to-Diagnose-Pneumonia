#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/christienatashiaarchie/Pneumonia-Detection/blob/master/Pneumonia_Detection.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# DIR Constants
cwd = os.getcwd()
TRAIN=cwd+'/PneumCovid/train/'
TEST=cwd+'/PneumCovid/test/'
VAL=cwd+'/PneumCovid/val/'


# In[1]:


model = tf.keras.models.Sequential([
  
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
  
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
  
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
  
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
  
    # The fifth convolution
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

  
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'), # 512 neuron hidden layer
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for ('pneumonia') clas and 1 for ('covid') class
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# In[6]:


model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics = ['accuracy'])


# In[7]:


train_datagen = ImageDataGenerator(rescale = 1/255)
test_datagen = ImageDataGenerator(rescale = 1/255)

train_generator = train_datagen.flow_from_directory(
    TRAIN,
    target_size = (300,300),
    batch_size = 163,
    color_mode="rgb",
    class_mode = 'binary'
)

validation_generator = test_datagen.flow_from_directory(
    TEST,
    target_size = (300, 300),
    batch_size = 156,
    color_mode="rgb",
    class_mode = 'binary'
)


# In[8]:


history = model.fit(
    train_generator,
    steps_per_epoch = 10,
    epochs = 10,
    validation_data = validation_generator
)


# In[9]:


fig, ax = plt.subplots(1,2)
fig.set_size_inches(10,5)

metric = ['accuracy', 'loss']
for i, m in enumerate(metric):
  ax[i].plot(history.history[m])
  ax[i].plot(history.history['val_'+ m])
  ax[i].set_title('Model {}'.format(m))
  ax[i].set_xlabel('epochs')
  ax[i].set_ylabel('m')
  ax[i].legend(['train', 'validation'])
plt.savefig('plot.png', dpi=300, bbox_inches='tight')


# In[10]:


# load new unseen dataset
test_datagen = ImageDataGenerator(rescale = 1/255)

test_generator = test_datagen.flow_from_directory(
    VAL,
    target_size = (300, 300),
    batch_size = 1, 
    color_mode="rgb",
    class_mode = 'binary'
)

eval_result = model.evaluate(test_generator)
print('loss rate at evaluation data :', eval_result[0])
print('accuracy rate at evaluation data :', eval_result[1])


# In[12]:


tfile = 'COVID19(575).jpg'
path = 'PneumCovid/predict/'+tfile

img = image.load_img(path, target_size=(300,300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis =0)

images = np.vstack([x])
classes = model.predict(images, batch_size = 10)
print(classes[0])
if classes[0]> 0.5:
    print(tfile + ' is pneumonia')
    plt.imshow(img)
else:
    print(tfile + ' is covid19 pneumonia')
    plt.imshow(img)


# In[ ]:




