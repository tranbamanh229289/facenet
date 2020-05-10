from os import listdir
from PIL import Image
import numpy
from numpy import asarray
from numpy import savez_compressed
from tensorflow.keras.models import load_model

data=numpy.load('/home/face_id/Facenet/data.npz')
train_x,train_y,test_x,test_y=data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']
model = load_model('/home/face_id/Facenet/facenet_keras.h5')

def get_embedding (model , face_pixels ):
    face_pixels = face_pixels.astype('float32')
    mean = face_pixels.mean()
    std=face_pixels.std()
    face_pixels = (face_pixels-mean)/std
    samples= numpy.expand_dims(face_pixels,axis=0)
    a= model.predict(samples)
    return a[0]
new_trainX=list()
new_testX=list()
for face_pixels in train_x:
    embedding = get_embedding(model, face_pixels)
    new_trainX.append(embedding)
for face_pixels in test_x:
    embedding = get_embedding(model,face_pixels)
    new_testX.append(embedding)

new_trainX = asarray(new_trainX)
new_testX=asarray(new_testX)
savez_compressed ('/home/face_id/Facenet/data1.npz',new_trainX,train_y,new_testX,test_y)