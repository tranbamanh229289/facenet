from os import listdir
from PIL import Image
import numpy
from numpy import asarray
from numpy import savez_compressed
from matplotlib import pyplot as plt
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

data= numpy.load('/home/face_id/Facenet/data1.npz')
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