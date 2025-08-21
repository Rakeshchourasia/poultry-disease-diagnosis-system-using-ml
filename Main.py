from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import cv2
import pickle
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import os
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.models import Sequential, load_model, Model
from keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.applications import InceptionV3
from keras.applications import MobileNetV2
from keras.applications import VGG16


main = tkinter.Tk()
main.title("Poultry Diseases Diagnostics Models using Deep Learning")
main.geometry("1300x1200")

global filename, labels
global X_train, y_train, X_test, y_test, labels, X, Y, vgg_model
global accuracy, precision, recall, fscore

def getLabel(name):
    global labels
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def uploadDataset():
    text.delete('1.0', END)
    global filename, dataset, labels, X, Y
    labels = []
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename)
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            if name not in labels:
                labels.append(name.strip())
    if os.path.exists("model/X.npy"):
        X = np.load('model/X.npy')
        Y = np.load('model/Y.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (80, 80))
                    X.append(img)
                    label = getLabel(name)
                    Y.append(label)   
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)                    
    text.insert(END,"Dataset Loading Completed\n")
    text.insert(END,"Total images found in dataset = "+str(X.shape[0])+"\n")
    text.insert(END,"Various Poultry Diseases found in Dataset : "+str(labels)+"\n\n")
    label, count = np.unique(Y, return_counts = True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (4, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def imagePreprocessing():
    global X, Y
    text.delete('1.0', END)
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Dataset Shuffling & Normalization Completed")

def splitDataset():
    global X, Y
    global X_train, y_train, X_test, y_test
    text.delete('1.0', END)
    #split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    text.insert(END,"Dataset Train & Test Split Details\n")
    text.insert(END,"80% dataset for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset for testing  : "+str(X_test.shape[0])+"\n")

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    global labels
    global accuracy, precision, recall, fscore
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    fig, axs = plt.subplots(1,2,figsize=(10, 3))
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axs[0]);
    ax.set_ylim([0,len(labels)])
    axs[0].set_title(algorithm+" Confusion matrix") 

    random_probs = [0 for i in range(len(testY))]
    p_fpr, p_tpr, _ = roc_curve(testY, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange',label="True classes")
    ns_fpr, ns_tpr, _ = roc_curve(testY, predict, pos_label=1)
    axs[1].plot(ns_tpr, ns_fpr, linestyle='--', label='Predicted Classes')
    axs[1].set_title(algorithm+" ROC AUC Curve")
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive rate')
    plt.tight_layout()
    plt.show()    
    
def runinceptionv3():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    inceptionv3 = InceptionV3(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
    for layer in inceptionv3.layers:
        layer.trainable = False
    headModel = inceptionv3.output
    headModel = AveragePooling2D(pool_size=(1, 1))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.3)(headModel)
    headModel = Dense(y_train.shape[1], activation="softmax")(headModel)
    inceptionv3_model = Model(inputs=inceptionv3.input, outputs=headModel)
    inceptionv3_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/inceptionv3_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/inceptionv3_weights.hdf5', verbose = 1, save_best_only = True)
        hist = inceptionv3_model.fit(X_train, y_train, batch_size = 64, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/inceptionv3_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        inceptionv3_model.load_weights("model/inceptionv3_weights.hdf5")
    predict = inceptionv3_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("InceptionV3", y_test1, predict)
    
def runMobilenet():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore, mobilenet_model
    mobilenet = MobileNetV2(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
    for layer in mobilenet.layers:
        layer.trainable = False
    headModel = mobilenet.output
    headModel = AveragePooling2D(pool_size=(1, 1))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.3)(headModel)
    headModel = Dense(y_train.shape[1], activation="softmax")(headModel)
    mobilenet_model = Model(inputs=mobilenet.input, outputs=headModel)
    mobilenet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/mobilenet_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/mobilenet_weights.hdf5', verbose = 1, save_best_only = True)
        hist = mobilenet_model.fit(X_train, y_train, batch_size = 64, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/mobilenet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        mobilenet_model.load_weights("model/mobilenet_weights.hdf5")
    predict = mobilenet_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("MobileNet", y_test1, predict)

def runVGG():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore, vgg_model
    vgg = VGG16(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
    for layer in vgg.layers:
        layer.trainable = False
    headModel = vgg.output
    headModel = AveragePooling2D(pool_size=(1, 1))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.3)(headModel)
    headModel = Dense(y_train.shape[1], activation="softmax")(headModel)
    vgg_model = Model(inputs=vgg.input, outputs=headModel)
    vgg_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/vgg_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/vgg_weights.hdf5', verbose = 1, save_best_only = True)
        hist = vgg_model.fit(X_train, y_train, batch_size = 64, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/vgg_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        vgg_model.load_weights("model/vgg_weights.hdf5")
    predict = vgg_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    calculateMetrics("VGG19", y_test1, predict)
    

def graph():
    #comparison graph between all algorithms
    df = pd.DataFrame([['InceptionV3','Accuracy',accuracy[0]],['InceptionV3','Precision',precision[0]],['InceptionV3','Recall',recall[0]],['InceptionV3','FSCORE',fscore[0]],
                       ['MobileNetV2','Accuracy',accuracy[1]],['MobileNetV2','Precision',precision[1]],['MobileNetV2','Recall',recall[1]],['MobileNetV2','FSCORE',fscore[1]],
                       ['VGG16','Accuracy',accuracy[2]],['VGG16','Precision',precision[2]],['VGG16','Recall',recall[2]],['VGG16','FSCORE',fscore[2]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(6, 3))
    plt.title("All Algorithms Performance Graph")
    plt.show()

def predictDisease():
    global vgg_model, labels
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (80, 80))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,80,80,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = vgg_model.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img, (700,400))
    cv2.putText(img, 'Poultry Disease Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Poultry Disease Predicted As : '+labels[predict], img)
    cv2.waitKey(0)

            

font = ('times', 16, 'bold')
title = Label(main, text='Poultry Diseases Diagnostics Models using Deep Learning',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Poultry Dataset", command=uploadDataset)
upload.place(x=10,y=500)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=400,y=500)

preprocessButton = Button(main, text="Preprocess Dataset", command=imagePreprocessing)
preprocessButton.place(x=10,y=550)
preprocessButton.config(font=font1)

splitButton = Button(main, text="Train & Test Split", command=splitDataset)
splitButton.place(x=330,y=550)
splitButton.config(font=font1)

inceptionButton = Button(main, text="Run InceptionV3 Algorithm", command=runinceptionv3)
inceptionButton.place(x=660,y=550)
inceptionButton.config(font=font1)

mobilenetButton = Button(main, text="Run MobileNetV2 Algorithm", command=runMobilenet)
mobilenetButton.place(x=10,y=600)
mobilenetButton.config(font=font1)

vggButton = Button(main, text="Run VGG16 Algorithm", command=runVGG)
vggButton.place(x=330,y=600)
vggButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=660,y=600)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Poultry Disease from Test Image", command=predictDisease)
predictButton.place(x=10,y=650)
predictButton.config(font=font1)



font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='magenta3')
main.mainloop()
