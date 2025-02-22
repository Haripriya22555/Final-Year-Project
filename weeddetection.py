from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from tkinter import ttk
from tkinter import filedialog
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout, Flatten
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os
import cv2
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import model_from_json

main = Tk()
main.title("Advancing Precision Agriculture:Deep Learning-Based Weed Species Identification Using Deep Weeds")
main.geometry("1300x1200")

global filename
global X, Y
global model
global accuracy

shapes = ['Negative', 'Snake weed', 'Rubber vine', 'Siam weed','Parkinsonia','Prickly acacia','Parthenium','Chinee apple','Lantana']
def getID(name):
    # Replace this with your implementation to assign labels based on the directory name
    # For example:
    if name == 'Negative':
        return 0
    elif name == 'Snake weed':
        return 1
    elif name == 'Rubber vine':
        return 2
    elif name == 'Siam weed':
        return 3 
    elif name == 'Parkinsonia':
        return 4
    elif name == 'Prickly acacia':
        return 5
    elif name == 'Parthenium':
        return 6
    elif name == 'Chinee apple':
        return 7
    elif name == 'Lantana':
        return 8
    else:              # Return an appropriate label for other cases
        return -1
def uploadDataset():
    global X, Y
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,'dataset loaded\n')

def imageProcessing():
    text.delete('1.0', END)
    global X, Y  
    '''
    
    X = []
    Y = []
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            print(name+" "+root+"/"+directory[j])
            if 'Thumbs.db' not in directory[j]:
                img = cv2.imread(root+"/"+directory[j])
                img = cv2.resize(img, (64,64))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(64,64,3)
                X.append(im2arr)
                Y.append(getID(name))
        
    X = np.asarray(X)
    Y = np.asarray(Y)
    print(Y)

    X = X.astype('float32')
    X = X/255    
    test = X[3]
    test = cv2.resize(test,(400,400))
    cv2.imshow("aa",test)
    cv2.waitKey(0)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    np.save('model/X.txt',X)
    np.save('model/Y.txt',Y)
    '''
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    text.insert(END,"Total number of images found in dataset is  : "+str(len(X))+"\n")
    text.insert(END,"Total disease found in dataset is : "+str(shapes)+"\n")
    

def close():
    main.destroy()
    text.delete('1.0', END)
    

font = ('times', 15, 'bold')
title = Label(main, text="Advancing Precision Agriculture:Deep Learning-Based Weed Species Identification Using Deep Weeds")
title.config(bg='powder blue', fg='olive drab')
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Image Processing & Normalization", command=imageProcessing)
processButton.place(x=20,y=150)
processButton.config(font=ff)


exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=20,y=450)
exitButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=85)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

main.config()
main.mainloop()
