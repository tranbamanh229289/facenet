from os import listdir
from PIL import Image
import numpy
from numpy import asarray
from numpy import savez_compressed
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model


#detection
def extract_face(filename ) :
    image = Image.open(filename)
    image = image.convert('RGB')
    pixel = asarray(image)
    image = Image.fromarray(pixel)
    image = image.resize((160,160))
    face_array = asarray(image)
    return face_array
def load_face(link):
    faces= list()
    for i in listdir(link ):
        path = link +i
        face= extract_face (path )
        faces.append(face)
    return faces
def load_data(link ):
    X=list()
    Y=list()
    for i in listdir (link ):
        path = link+i+'/'
        faces= load_face(path)
        label = [i for _ in range (len(faces))]
        X.extend(faces)
        Y.extend(label)
    return X,Y

link_train ='/home/face_id/Facenet/VN_train/'
link_test='/home/face_id/Facenet/VN_test/'
train_x ,train_y= load_data(link_train)
test_x,test_y=load_data(link_test)
savez_compressed('/home/face_id/Facenet/data.npz',train_x,train_y,test_x,test_y)
data=numpy.load('/home/face_id/Facenet/data.npz')