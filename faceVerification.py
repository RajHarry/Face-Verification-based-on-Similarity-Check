#================import Required Libraries=============
import tensorflow as tf
import numpy as np
import facenet
import cv2
import argparse
import time
import glob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score,recall_score,confusion_matrix as cm,precision_score,accuracy_score
import warnings
warnings.simplefilter("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
#=============Done importing Libraries=================
#===============Initialize Constants as in "Facenet Official Paper"====================
def get_facenet_model():
    return facenet_model_emb
#time library for time calculation.
#This model Takes Around "10sec" to load on my SYSTEM(with CPU).
#It May Various in System to System.(may fast load with GPU).

minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
input_image_size = 160


threshold = 0.90    # set yourself to meet your requirement            
#===============Done with "initializing Constants"====================
            
#=========================Load Model===================
#This is a pretrained model.
#This model is used to extract "face Features" from a Given Face
#You can get this model from Below Links
st = time.time()
facenet.load_model("trained_models/20170512-110547.pb")
en = time.time()
print('elapsed(model loading): ',en-st)
facenet_model_emb = facenet

sess = tf.Session() #create Tensorflow Session
# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
#======================Done Loading Model==============
def F(beta, precision, recall):
    return (beta*beta + 1)*precision*recall / (beta*beta*precision + recall)
#======================Start "Similarity Check(main)"==============
def getEmbedding(resized):
    """Generate 128 Embeddings for a image(prewhitened)"""
    reshaped = resized.reshape(-1,input_image_size,input_image_size,3)
    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    #sess.close()
    return embedding

def getFace(img):
    """Take image and give this to Facenet.image will be prewhitened to detect edges from a image."""
    faces = []
    #print("i'm here")
    prewhitened = facenet.prewhiten(img)
    faces.append({'face':img,'embedding':getEmbedding(prewhitened)})
    return faces
def compare2face(face1,face2):
    """Compare Two faces of a User"""
    # calculate Euclidean distance
    if face1 and face2: #if face1 and face2 has not empty(both faces are detected)
        cdist = cosine_similarity(face1[0]['embedding'], face2[0]['embedding'])
        dist = np.sqrt(np.sum(np.square(np.subtract(face1[0]['embedding'], face2[0]['embedding']))))
        return cdist,dist
    else:   #if photo is not detected from one of the picture(face1 or face2)
        print("photo is not detected")
        return -1
def similarity_check():
    #0.76,0411
    acc = []
    global l1,l2,actual,pred
    # some constants kept as default from facenet
    path2 = glob.glob('aligned_faces/verify/*')
    path1 = glob.glob("aligned_faces/base_image/*")
    path1.reverse()
    #print(path2)

    for single in path1:
        name = single.split('\\')[-1].split('_')[1].split('.')[0]

        img1 = cv2.imread(single)    #read image(original)
        face1 = getFace(img1)

        w_pred=0
        print("\nMain\ttemp\t\tEuclid\t\tCosine\t\tResult")
        for file in path2:
            img2 = cv2.imread(file)    #read image(test)
            face2 = getFace(img2)
            user_name = file.split('\\')[-1].split('_')[1].split('.')[0]
            print("user_name: ",user_name)
            print("name: ",name)
            if(int(user_name[:2]) == int(name)):
                l1.append(0)
            else:
                l1.append(1)
        
            st = time.time()
            c_dist,distance = compare2face(face1,face2) #to get Distance between faces(face1 and face2)
            en = time.time()
            #print("length of username: ",len(user_name))
            #print('elapsed time(comparing): ',en-st)
            if(threshold >= distance):
            	l2.append(0)
            else:
                l2.append(1)
                #print(user_name)
                print(name,"\t\t",user_name,'\t\t',str(distance)[:5]+"\t\t"+str(c_dist[0][0])[:5],'\t',("same person" if distance <= threshold else "not same person"))
        
        print('*****************************************')
        print("user_name: ",name)
        actual.extend(l1)
        pred.extend(l2)
        print("length_l1: ",len(l1),"length_l2: ",len(l2))
        prec = precision_score(l2,l1,average="macro")
        rec  = recall_score(l2,l1,average="macro")
        print("precision_score: ",prec)
        print("recall_score: ",rec)
        print("overall F1_score: ",F(1,prec,rec))
        print("overall F0.5_score: ",F(0.5,prec,rec))
        print("overall F2_score: ",F(2,prec,rec))
        print("total wrong predictions: ",w_pred)
        acc.append(F(2,prec,rec))
        print('*****************************************')
        l1,l2,single_f1 = [],[],[]
        break
    return acc

global actual,pred,l1,l2
actual,pred,l1,l2 = [],[],[],[]

acc = similarity_check()
'''
prec = precision_score(pred,actual,average="macro")
rec  = recall_score(pred,actual,average="macro")
print("precision_score: ",prec)
print("recall_score: ",rec)
print("overall F1_score: ",F(1,prec,rec))
print("overall F0.5_score: ",F(0.5,prec,rec))
print("overall F2_score: ",F(2,prec,rec))
print("total accuracies: ",acc)
print("average accuracies: ",sum(acc)/len(acc))
'''