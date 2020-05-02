from os import listdir
from PIL import Image
import numpy
from numpy import asarray
from numpy import savez_compressed
from mtcnn import MTCNN
from matplotlib import pyplot as plt
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

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

link_train ='C:/Users/ThinkKING/Downloads/VN_train/'
link_test='C:/Users/ThinkKING/Downloads/VN_test/'
train_x ,train_y= load_data(link_train)
test_x,test_y=load_data(link_test)
savez_compressed('C:/Users/ThinkKING/Downloads/data.npz',train_x,train_y,test_x,test_y)
data=numpy.load('C:/Users/ThinkKING/Downloads/data.npz')
train_x,train_y,test_x,test_y=data['arr_0'],data['arr_1'],data['arr_2'],data['arr_3']
model = load_model('C:/Users/ThinkKING/Downloads/facenet_keras.h5')

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
savez_compressed ('C:/Users/ThinkKING/Downloads/data1.npz',new_trainX,train_y,new_testX,test_y)

data= numpy.load('C:/Users/ThinkKING/Downloads/data1.npz')
train_x= data['arr_0']
train_y=data['arr_1']
test_x=data['arr_2']
test_y=data['arr_3']

in_encoder=Normalizer(norm='l2')
train_x= in_encoder.transform(train_x)
test_x=in_encoder.transform(test_x)
out_encoder=LabelEncoder()
out_encoder.fit(train_y)
train_y=out_encoder.transform(train_y)
test_y=out_encoder.transform(test_y)

model = SVC(kernel='linear',probability=True)
model.fit(train_x,train_y)

a_train=model.predict(train_x)
a_test=model.predict(test_x)

score_train=accuracy_score(train_y,a_train)
score_test=accuracy_score(test_y,a_test)

print ('Accuracy test  : ',score_train*100 )
print ('Accuracy test :',score_test *100)







