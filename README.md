# Facenet
## Data
* This is a face recognition problem with the VN -ceb dataset with 13000 photos of 1020 people celebrities .The link is here [VN_ceb](https://viblo.asia/p/vn-celeb-du-lieu-khuon-mat-nguoi-noi-tieng-viet-nam-va-bai-toan-face-recognition-Az45bG9VKxY)
* Each person will have k photos. I divide 80% of the photos of each person into a train set, the rest become a test data set
## Code 
* I use mtCNN model to dectect the face in every photos .Then I use the keras library to extract the features.Finally, I use the SVM algorithm to classify images from features. Use that model for the train data set.
## Tutorial :
* [link tutorial (https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/?fbclid=IwAR2DImNtGHDirKqFkVu4Jd9AZIsjyv2KsXAF0IG6-qwqAJJYj0GuKL9vrpI)
