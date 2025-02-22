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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score,recall_score
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
from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform

main = Tk()
main.title("Advancing Precision Agriculture:Deep Learning-Based Weed Species Identification Using Deep Weeds")
main.geometry("1300x1200")

global filename
global X, Y
global model
global accuracy

shapes = ['Negative', 'Snake weed', 'Rubber vine', 'Siam weed','Prickly acacia','Parthenium','Chinee apple','Lantana']
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
def svmImagePreprocessing():
    global x_test,x_train,y_test,y_train,df
    Categories=['Negative','Snake weed']
    
    flat_data_arr=[] #input array
    target_arr=[] #output array
    datadir=r"data"
    #create file paths by combining the datadir (data directory) with the filenames 'flat_data.npy
    flat_data_file = os.path.join(datadir, 'flat_data.npy')
    target_file = os.path.join(datadir, 'target.npy')
    if os.path.exists(flat_data_file) and os.path.exists(target_file):
        # Load the existing arrays
        flat_data = np.load(flat_data_file)
        target = np.load(target_file)
        #dataframe
        df=pd.DataFrame(flat_data)
        df['Target']=target #associated the numerical representation of the category (index) with the actual image data
        df
        #input data
        
        x=df.iloc[:,:-1]
        #output data
        y=df.iloc[:,-1]
        x

        # Splitting the data into training and testing sets
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.40,random_state=77)
        
        text.insert(END,"Total number of images found in dataset is  : "+str(np.shape(x))+"\n")
        text.insert(END,"Total species found in dataset is : "+str(np.shape(y))+"\n")
        plt.figure(figsize=(8, 6))
        ax = sns.countplot(data=df, x='Target')
        plt.xlabel('Target', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Count Plot for Target', fontsize=14)
        # Add count labels on top of the bars
        for p in ax.patches:
                ax.annotate(f"{p.get_height()}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    
        plt.show()
    else:
        #path which contains all the categories of images
        for i in Categories:
            print(f'loading... category : {i}')
            path=os.path.join(datadir,i)
            #create file paths by combining the datadir (data directory) with the i
            for img in os.listdir(path):
                img_array=imread(os.path.join(path,img))#Reads the image using imread.
                img_resized=resize(img_array,(150,150,3)) #Resizes the image to a common size of (150, 150, 3) pixels.
                flat_data_arr.append(img_resized.flatten()) #Flattens the resized image array and adds it to the flat_data_arr.
                target_arr.append(Categories.index(i)) #Adds the index of the category to the target_arr.
                #this index is being used to associate the numerical representation of the category (index) with the actual image data. This is often done to provide labels for machine learning algorithms where classes are represented numerically. In this case, 'ORGANIC' might correspond to label 0, and 'NONORGANIC' might correspond to label 1.
                print(f'loaded category:{i} successfully')
                #After processing all images, it converts the lists to NumPy arrays (flat_data and target).
                flat_data=np.array(flat_data_arr)
                target=np.array(target_arr)
        # Save the arrays(flat_data ,target ) into the files(flat_data.npy,target.npy)
        np.save(os.path.join(datadir, 'flat_data.npy'), flat_data)
        np.save(os.path.join(datadir, 'target.npy'), target)
def svmModel():
    global model,x_train,y_train
    global accuracy
    text.delete('1.0', END)
    if os.path.exists('model/SVM_Classifier.pkl'):
        with open('model/SVM_Classifier.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
            y_pred=model.predict(x_test)
            acc=accuracy_score(y_test,y_pred)
            text.insert(END,"SVM Model Accuracy = "+str(acc)+"\n")
            precision=precision_score(y_test,y_pred)
            text.insert(END,"SVM Model precision = "+str(precision)+"\n")
            recall=recall_score(y_test,y_pred)
            text.insert(END,"SVM Model recall = "+str(recall)+"\n")
            cm=confusion_matrix(y_test,y_pred)
            text.insert(END,"SVM Model confusion_matrix = "+str(cm)+"\n")
            report=classification_report(y_test,y_pred)
            text.insert(END,"SVM Model classification_report = "+"\n\n"+str(report)+"\n")
    else:
        model = SVC(kernel='linear', C=1.0, random_state=42)
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        Acc=accuracy_score(y_test,y_pred)*100
        print("Accuracy",Acc)
        # Dump the trained Naive Bayes classifier with Pickle
        filename = 'SVM_Classifier.pkl'
        # Open the file to save as pkl file
        SVM_Model_pkl = open(filename, 'wb')
        #when you use 'wb' as the mode when opening a file, you are telling Python to open the file in write mode and treat it as a binary file. 	This is commonly used when saving non-textual data, such as images, audio, or serialized objects like machine learning models
        pickle.dump(model, SVM_Model_pkl)
        #function to serialize and save the rf object (which is your trained Random Forest model) into the Pickle file opened as RF_Model_pkl.
        # Close the pickle instances
        SVM_Model_pkl.close()
        text.insert(END,"SVM Model Accuracy = "+str(Acc))

    
def imageProcessing():
    text.delete('1.0', END)
    global X, Y,Xtrain1,Xtest1,Ytrain1,Ytest1
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
    text.insert(END,"Total species found in dataset is : "+str(shapes)+"\n")
    Xtrain1,Xtest1,Ytrain1,Ytest1=train_test_split(X,Y,test_size=0.30,random_state=77)
   
def cnnModel():
    global model
    global accuracy
    text.delete('1.0', END)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        json_file.close()    
        model.load_weights("model/model_weights.h5")
        model._make_predict_function()   
        print(model.summary())
        f = open('model/history.pckl', 'rb')
        accuracy = pickle.load(f)
        f.close()
        acc = accuracy['accuracy']
        acc = acc[9] * 100
        text.insert(END,"CNN  Model Accuracy = "+str(acc))
        
    else:
        model = Sequential() #resnet transfer learning code here
        model.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Convolution2D(32, 3, 3, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Flatten())
        model.add(Dense(output_dim = 256, activation = 'relu'))
        model.add(Dense(output_dim = 9, activation = 'softmax'))
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        print(model.summary())
        hist = model.fit(X, Y, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
        model.save_weights('model/model_weights.h5')            
        model_json = model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        accuracy = pickle.load(f)
        f.close()
        acc = accuracy['accuracy']
        acc = acc[9] * 100
        text.insert(END,"CNN  Model Prediction Accuracy = "+str(acc))
        
def predict():
    global model
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    test = np.asarray(im2arr)
    test = test.astype('float32')
    test = test/255
    preds = model.predict(test)
    predict = np.argmax(preds)
    img = cv2.imread(filename)
    img = cv2.resize(img, (500,500))
    cv2.putText(img, 'Weed Identified as : '+shapes[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imshow('Weed Identified as : '+shapes[predict], img)
    cv2.waitKey(0)
    
def graph():
    acc = accuracy['accuracy']
    loss = accuracy['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/Loss')
    plt.plot(acc, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('Iteration Wise Accuracy & Loss Graph')
    plt.show()
    
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

processButton = Button(main, text="SVM Image processing", command=svmImagePreprocessing)
processButton.place(x=20,y=200)
processButton.config(font=ff)

svmButton = Button(main, text="Build SVM Model", command=svmModel)
svmButton.place(x=20,y=250)
svmButton.config(font=ff)

modelButton = Button(main, text="Build CNN Model", command=cnnModel)
modelButton.place(x=20,y=300)
modelButton.config(font=ff)

predictButton = Button(main, text="Upload Test Image & Identify ", command=predict)
predictButton.place(x=20,y=350)
predictButton.config(font=ff)

graphButton = Button(main, text="Accuracy & Loss Graph", command=graph)
graphButton.place(x=20,y=400)
graphButton.config(font=ff)

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
