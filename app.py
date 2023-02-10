# Importing necessities

import os
import shutil
import numpy as np
import PIL
import PIL.Image, PIL.ImageTk
import tkinter as tk
import cv2 as cv
import camera
import model
import model1
import model2

# Boolean to track whether to move onto second window where the model is trained, predictions made, etc.
cont = True

# Variables and constants
NUMCLASSES = 3
WIDTH = 150
HEIGHT = 150
predictCounter = 1

class App:
    def __init__(self):

        # Sets up file paths for data set folder and prediction set folder

        #self.window = window
        #self.windowTitle = window_title
        self.dataDir = 'data'
        self.predictDir = 'predictions'
        if os.path.exists(self.dataDir):
            shutil.rmtree(self.dataDir)
        if os.path.exists(self.predictDir):
            shutil.rmtree(self.predictDir)
        os.mkdir(self.dataDir)
        os.mkdir(f'{self.dataDir}/1')
        os.mkdir(f'{self.dataDir}/2')
        os.mkdir(f'{self.dataDir}/3')
        os.mkdir(self.predictDir)

        # Keeps track of the number of images added to each class's data
        self.counters = [1, 1, 1]

        #self.model =  model1.Model1()
        self.model = None

        self.toggle = False

        self.camera = camera.Camera()

        self.classNames = self.setupGUI()

        # Storing the class names in variables
        if self.classNames != []:
            first = self.classNames[0]
            second = self.classNames[1]
            third = self.classNames[2]

        # If the user did not close the first window and wants to proceed,
        if cont:
            # Sets up the new window
            interface = tk.Tk()
            # width and height come from camera: width = self.camera.width, height = self.camera.height
            interface.geometry(str(int(self.camera.width))+"x"+str(int(self.camera.height+160)))
            #interface.geometry("1200x600")
            label = tk.Label(interface)
            label.grid(row=0, column=0)
            cap = cv.VideoCapture(0)

            interface.title("Image Classification")

            # Buttons

            #toggleBtn = tk.Button(interface, text="Auto Prediction", width=50, command=self.autoToggle) #command = self.autoToggle
            firstBtn = tk.Button(interface, text=first, width=50, command=lambda: self.saveForClass(1)) #command = lambda: self.saveForClass()
            secondBtn = tk.Button(interface, text=second, width=50, command=lambda: self.saveForClass(2)) #command = lambda: self.saveForClass()
            thirdBtn = tk.Button(interface, text=third, width=50, command=lambda: self.saveForClass(3)) #command = lambda: self.saveForClass()
            trainBtn = tk.Button(interface, text="Train Model", width=50, command=self.createModel)
            predictBtn = tk.Button(interface, text="Predict", width=50, command=self.predict)
            resetBtn = tk.Button(interface, text="Reset", width=50, command = self.reset)

            firstBtn.place(x = self.camera.width//2 - 100, y = 490+20, width = 200, height = 20)
            secondBtn.place(x = self.camera.width//2 - 100, y = 490+40, width = 200, height = 20)
            thirdBtn.place(x = self.camera.width//2 - 100, y = 490+60, width = 200, height = 20)
            trainBtn.place(x = self.camera.width//2 - 100, y = 490+80, width = 200, height = 20)
            predictBtn.place(x = self.camera.width//2 - 100, y = 490+100, width = 200, height = 20)
            resetBtn.place(x = self.camera.width//2 - 100, y = 490, width = 200, height = 20)
                                                                                                                                                                                    

            # Helper method to show the webcam
            def show_frames():
                # Get the latest frame and convert into Image
                cvimage= self.camera.getNextFrame()[1]
                img = PIL.Image.fromarray(cvimage)
                # Convert image to PhotoImage
                imgtk = PIL.ImageTk.PhotoImage(image = img)
                label.imgtk = imgtk
                label.configure(image=imgtk)
                # Repeat after an interval to capture continiously
                label.after(20, show_frames)

            show_frames()

            interface.mainloop()

    # When the button for a certain class is clicked, the current frame gets stored into that class's folder
    def saveForClass(self, classNum):
        # Gets the frame
        returnval, frame = self.camera.getNextFrame()
        
        # Stores it in the folder
        cv.imwrite(f'{self.dataDir}/{classNum}/frame{self.counters[classNum-1]}.jpg', cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        img = PIL.Image.open(f'{self.dataDir}/{classNum}/frame{self.counters[classNum-1]}.jpg')
        img.thumbnail((WIDTH, HEIGHT), PIL.Image.ANTIALIAS)
        img.save(f'{self.dataDir}/{classNum}/frame{self.counters[classNum-1]}.jpg')
        
        # Updates the counter
        self.counters[classNum-1] += 1

    # When the train button is clicked, the model object from the Model2 class is created
    def createModel(self):
        self.model = model2.Model2(self.classNames, self.dataDir)

    # Resets the data set to be empty
    def reset(self):
        for dir in ['1', '2', '3']:
            for file in os.listdir(f"data/{dir}"):
                filePath = os.path.join(f"data/{dir}", file)
                if os.path.isfile(filePath):
                    os.unlink(filePath)

        self.counters = [1, 1, 1]
        #self.model = model.Model()
        #self.classLabel.config(text="Class")
        print("Data reset")

    # Used for auto prediction
    def update(self):
        if self.autoPredict:
            self.predict()
            
        returnval, frame = self.camera.getNextFrame()

        if returnval:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0,0,image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    
    def predict1(self):
        global predictCounter
        
        frame = self.camera.getNextFrame()
        #prediction = self.model.predict(frame)

        cv.imwrite(f'{self.predictDir}/frame{predictCounter}.jpg', cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        img = PIL.Image.open(f'{self.predictDir}/frame{predictCounter}.jpg')
        img.thumbnail((WIDTH, HEIGHT), PIL.Image.ANTIALIAS)
        img.save(f'{self.predictDir}/frame{predictCounter}.jpg')

        newImg = model1.preprocessNewImage(img, f'{self.predictDir}/frame{predictCounter}.jpg')
        predictions = self.model.predict(newImg)
        
        predictedClass = self.classNames[np.argmax(predictions[0])]
        print(f"Prediction is: {predictedClass}")

        '''
        if prediction == 1:
            self.classLabel.config(text=first)
            print(first)
            return first
        elif prediction == 2:
            self.classLabel.config(text=second)
            print(first)
            return second
        elif prediction == 3:
            self.classLabel.config(text=second)
            print(first)
            return second
        '''
        predictCounter += 1

    # When the predict button is clicked, the predict method of the Model2 class runs
    def predict(self):
        returnval, image = self.camera.getNextFrame()

        self.model.makePrediction(image)
    
    # Sets up the GUI
    def setupGUI(self):
        classNames = []
        def getClasses():
            if firstclass.get() == "" or secondclass.get() == "" or thirdclass.get() == "":
                print("Must enter a class")
            elif firstclass.get() == secondclass.get() or secondclass.get() == thirdclass.get() or firstclass.get() == thirdclass.get():
                print("Must have two distinct classes")
            else:

                classNames.append(firstclass.get())
                classNames.append(secondclass.get())
                classNames.append(thirdclass.get())

                classSetup.destroy()

        # Closes the window properly
        def onClosing():
            global cont
            cont = False
            classNames = []
            classSetup.destroy()

        classSetup = tk.Tk()

        classSetup.geometry("500x220")
        classSetup.title("Image Classification")

        label = tk.Label(classSetup, text = "Enter the classes to use for classification:", font = ('Arial', 18))
        label.pack(padx=20, pady=20)

        
        firstclass = tk.Entry(classSetup)
        firstclass.pack()

        secondclass = tk.Entry(classSetup)
        secondclass.pack(pady=10)

        thirdclass = tk.Entry(classSetup)
        thirdclass.pack()

        confirm = tk.Button(classSetup, text="Confirm", command=getClasses, width=15)
        confirm.pack(pady=20)

        classSetup.protocol("WM_DELETE_WINDOW", onClosing)
        classSetup.mainloop() ###

        return classNames
    
    def __del__(self):
        shutil.rmtree(self.dataDir)
        shutil.rmtree(self.predictDir)