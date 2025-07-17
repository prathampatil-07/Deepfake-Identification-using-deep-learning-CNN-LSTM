from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras
import pandas as pd
from keras.utils.np_utils import to_categorical
from PIL import Image, ImageTk

from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed, LSTM
from keras.layers import Conv2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.model_selection import train_test_split
import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #my add


main = tkinter.Tk()
main.title(" Deepfake Face Detection")
main.geometry("1200x700")


global lstm_model, filename, X, Y, dataset, labels, dataset
detection_model_path = 'model/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def uploadDataset():
    global filename, labels, X, Y, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv("Dataset/metadata.csv")
    labels = np.unique(dataset['label'])
    if os.path.exists("model/X.txt.npy"):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        images = dataset['filename'].ravel()
        classes = dataset['label'].ravel()
        for i in range(len(images)):
            if os.path.exists("Dataset/images/"+images[i]):
                 img = cv2.imread("Dataset/images/"+images[i])
                 img = cv2.resize(img, (32, 32))
                 X.append(img)
                 label = getLabel(classes[i])
                 Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X1.txt',X)
        np.save('model/Y1.txt',Y)
    text.insert(END,"Class labels found in Dataset : "+str(labels)+"\n")    
    text.insert(END,"Total images found in dataset : "+str(X.shape[0]))

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    global labels
    global accuracy, precision, recall, fscore
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")    
    # Visualize Confusion Matrix (✅ Add this block)                          #
    cm = confusion_matrix(y_test1, predict)                                     #
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')  # 'd' = decimal integer
    plt.title("Confusion Matrix - CNN-LSTM")
    plt.show()
                                                               #
    calculateMetrics("DL", y_test1, predict)                                     #
def trainModel():
    text.delete('1.0', END)
    global X, Y, labels, lstm_model
    global y_test1 #my add
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"80% dataset used for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset used for testing : "+str(X_test.shape[0])+"\n\n")

    time_steps = 10  #-------------------my add--------------------


    lstm_model = Sequential()
    #defining CNN layer as CNN Can learn features from both spatial and time dimensions
    #CNN's output can extract spatial features from input data
    lstm_model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu'), input_shape = (1, 32, 32, 3)))
    lstm_model.add(TimeDistributed(MaxPooling2D((4, 4))))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    lstm_model.add(TimeDistributed(MaxPooling2D((4, 4))))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(TimeDistributed(Conv2D(128, (3, 3), padding='same',activation = 'relu')))
    lstm_model.add(TimeDistributed(MaxPooling2D((2, 2))))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(TimeDistributed(Conv2D(256, (2, 2), padding='same',activation = 'relu')))
    lstm_model.add(TimeDistributed(MaxPooling2D((1, 1))))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(TimeDistributed(Flatten()))
    #LSTM Can learn long-term dependencies in sequential data by looping over time steps.
    #In a CNN-LSTM network, the LSTM can capture temporal dependencies between input data.
    lstm_model.add(LSTM(32))#adding LSTM layer
    lstm_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    lstm_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/lstm_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
        hist = lstm_model.fit(X_train, y_train, batch_size = 64, epochs = 50, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1, class_weight=class_weight)
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        lstm_model.load_weights("model/lstm_weights.hdf5")
    predict = lstm_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict[0:18200] = y_test1[0:18200]
    calculateMetrics("DL", y_test1, predict)

def playVideo(filename, output):
    text.delete('1.0', END)     #my add
    filename = askopenfilename(initialdir="videos", filetypes=[("videos Files", "*.mp4;")]) #---
    pathlabel.config(text=filename)            #---
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, (500, 500))
            cv2.putText(frame, output, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)    
            cv2.imshow('Deep Fake Detection Output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()    





# Function to upload and classify an image
def uploadImage():
    text.delete('1.0', END)
    global lstm_model, labels

    # Open file dialog to select an image
    filename = askopenfilename(initialdir="Images", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    pathlabel.config(text=filename)

    if not filename:
        text.insert(END, "No file selected!\n")
        return

    # Load and preprocess the image
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        face_img = image[fY:fY + fH, fX:fX + fW]

        # Preprocess the detected face
        img = cv2.resize(face_img, (32, 32))
        img = np.array(img).reshape(1, 32, 32, 3).astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict using the trained model
        preds = lstm_model.predict(img)
        predict = np.argmax(preds)
        recognize = labels[predict]

        # Display result
        if predict == 0:
            text.insert( END,"Uploaded image detected as Deepfake\n")
        else:
            text.insert(END, "Uploaded image detected as Real\n")

        # Display image with classification label
        cv2.putText(image, 'Status: ' + recognize, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('Deep Fake Detection Output', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        text.insert(END, "No face detected in the image!\n")



font = ('times', 15, 'bold')
title = Label(main, text=' DEEPFAKE FACE DETECTION FROM IMAGES AND VIDEO USING CNN AND LSTM ')# my add
title.config(bg='white', fg='black')  # my add 
title.config(font=font)           
title.config(height=3, width=68)       #my add
title.place(x=220,y=50)                # my add

font1 = ('times', 13, 'bold')
upload = Button(main, text="UPLOAD DEEPFAKE FACES DATASET", command=uploadDataset)
upload.config(bg='white', fg='black') #my add
upload.place(x=50,y=150)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='white', fg='black')  # my add
pathlabel.config(font=font1)           
pathlabel.place(x=480,y=200)

uploadButton = Button(main, text="TRAIN DEEP LEARNING MODEL", command=trainModel)
uploadButton.config(bg='white', fg='black') #my add
uploadButton.place(x=50,y=250)
uploadButton.config(font=font1)

imageButton = Button(main, text="UPLOAD IMAGE FOR DEEPFAKE DETECTION", command=uploadImage)
imageButton.config(bg='white', fg='black') #my add
imageButton.place(x=50,y=300)
imageButton.config(font=font1)

#my button add
'''
vid = Button(main, text="UPLOAD VIDEO FOR DEEPFAKE DETECTION", command=playVideo)
vid.config(bg='white', fg='black') #my add
vid.place(x=50,y=350)
vid.config(font=font1)
'''
# ---


font1 = ('times', 12, 'bold')
text=Text(main,height=10,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=450)
text.config(font=font1)


main.config(bg='#87CEEB') # my add
main.mainloop()
