from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
from keras.models import model_from_json
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing import image
import os
from numpy import dot
from numpy.linalg import norm
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import imutils
import nltk


main = tkinter.Tk()
          # ANALYSING SENTIMENTAL REVIEWS OF E-GOVERNMENT SERVICES USING DEEP LEARNING
main.title("Analysing Sentimental Reviews of E-Government Services using Deep Learning")
main.geometry("1350x500")

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
NEGATIVE_EMOTIONS = ["angry", "disgust", "scared", "sad"]
POSITIVE_EMOTIONS = ["happy", "surprised", "neutral"]
POSITIVE_RESULT = 'It is a good government poilcy/service.'
NEGATIVE_RESULT = 'Very less positive feedbacks, government should consider taking action on it'

global filename
global text_sentiment_model
global face_detection
global image_sentiment_model
global digits_cnn_model
models_created = []


def digitModel():
    if 'digitModel' not in models_created:
        global digits_cnn_model
        with open('models/digits_cnn_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            digits_cnn_model = model_from_json(loaded_model_json)

        digits_cnn_model.load_weights("models/digits_cnn_weights.h5")
        digits_cnn_model._make_predict_function()
        print(digits_cnn_model.summary())
        text.delete('1.0', END)
        text.insert(END, 'Digits based Deep Learning CNN Model generated\n')

        models_created.append('digitModel')
    else:
        text.delete('1.0', END)
        text.insert(END, 'Digits based Deep Learning CNN Model is already generated\n')


def sentimentModel():
    if 'sentimentModel' not in models_created:
        text.delete('1.0', END)

        global text_sentiment_model
        global image_sentiment_model
        global face_detection

        text_sentiment_model = joblib.load('models/sentimentModel.pkl')
        text.insert(END, 'Text based sentiment Deep Learning CNN Model generated\n')

        face_detection = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        image_sentiment_model = load_model('models/_mini_XCEPTION.106-0.65.hdf5', compile=False)
        text.insert(END, 'Image based sentiment Deep Learning CNN Model generated\n')
        print(image_sentiment_model.summary())

        models_created.append('sentimentModel')
    else:
        text.delete('1.0', END)
        text.insert(END, 'Text & Image based sentiment Deep Learning CNN Model is already generated\n')


def digitRecognize():
    try:
        # to check whether digits_cnn_model is been generated
        digits_cnn_model

        text.delete('1.0', END)
        global filename
        filename = filedialog.askopenfilename(initialdir="testImages")
        # pathlabel.config(text=filename)
        # text.insert(END, filename + " loaded\n")

        imagetest = image.load_img(filename, target_size=(28, 28), grayscale=True)
        imagetest = image.img_to_array(imagetest)
        imagetest = np.expand_dims(imagetest, axis=0)
        pred = digits_cnn_model.predict(imagetest.reshape(1, 28, 28, 1))
        predicted = str(pred.argmax())
        imagedisplay = cv2.imread(filename)
        gorig = imagedisplay.copy()
        output = imutils.resize(gorig, width=400)
        cv2.putText(output, "Digits Predicted As : " + predicted, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Predicted Image Result", output)
        cv2.waitKey(0)
    except NameError:
        text.delete('1.0', END)
        text.insert(END, "Please Generate the Digits Recognition Model first.")


def takeOpinion():
    user = simpledialog.askstring("Please enter your name", "Username")
    # appended this policy to the string which we write into the file
    policy = simpledialog.askstring("Please enter Government Policy name related to your opinion", "Please enter Government Policy name related to your opinion")
    opinion = simpledialog.askstring("Government Service Opinion", "Please write your Opinion about government services & policies")

    # use # opinion # policy
    f = open("Peoples_Opinion/opinion.txt", "a+")
    f.write(user + "#" + opinion + "#" + policy + "\n")
    f.close()
    messagebox.showinfo("Thank you for your opinion", "Your opinion saved for reviews")


def stem(textmsg):
    stemmer = nltk.stem.PorterStemmer()
    textmsg_stem = ''
    textmsg = textmsg.strip("\n")
    words = textmsg.split(" ")
    words = [stemmer.stem(w) for w in words]
    textmsg_stem = ' '.join(words)
    return textmsg_stem


def viewSentiment():
    try:
        # to check whether text_sentiment_model is been generated
        text_sentiment_model

        text.delete('1.0', END)
        with open("Peoples_Opinion/opinion.txt", "r") as file:
            for line in file:
                line = line.strip('\n')
                line = line.strip()
                arr = line.split("#")
                text_processed = stem(arr[1])
                X = [text_processed]
                sentiment = text_sentiment_model.predict(X)
                predicts = 'None'
                if sentiment[0] == 0:
                    predicts = "Negative"
                if sentiment[0] == 1:
                    predicts = "Positive"
                text.insert(END, "Username : " + arr[0] + "\n");
                text.insert(END, "Policy/Service : " + arr[2] + "\n");
                text.insert(END, "Opinion  : " + arr[1] + " : Sentiment Detected As : " + predicts + "\n\n")
    except NameError:
        text.delete('1.0', END)
        text.insert(END, "Please Generate the Image and Text based sentimental Model first.")


def uploadPhotoAndDetectSentiment():
    try:
        # to check if we have generated the model
        face_detection

        filename = filedialog.askopenfilename(initialdir="expression_images_to_upload")
        user = simpledialog.askstring("Please enter your name", "Username")
        policy = simpledialog.askstring("Please enter Government Policy name related to Facial Expression", "Please enter Government Policy name related to Facial Expression")

        text.delete('1.0', END)
        text.insert(END, "Processing your image...\n\n")
        main.update_idletasks()

        img = cv2.imread(filename)
        # writePath = "sentimentImages/" + user + "-" + policy + ".jpg"

        # cv2.imwrite(writePath, img)

        # use img to Detect the sentiment
        faces = face_detection.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        msg = ''
        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (x, y, w, h) = faces
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = temp[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = image_sentiment_model.predict(roi)[0]
            emotion_probability = np.max(preds)
            sentiment = EMOTIONS[preds.argmax()]

            writePath = "sentimentImages/" + user + "-" + policy + "-" + sentiment + ".jpg"

            # image has been saved into sentimentImages
            cv2.imwrite(writePath, img)

            msg = "Sentiment detected as : " + sentiment
            text.insert(END, msg)
            main.update_idletasks()

            img_height, img_width = img.shape[:2]
            cv2.putText(img, msg, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow(writePath, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except NameError:
        text.delete('1.0', END)
        text.insert(END, "Please Generate the Text & Image Based Sentiment Model first.")

    # use this logic for Consolicated Result
    # filename = 'sentimentImages'
    # for root, dirs, files in os.walk(filename):
    #     for fdata in files:
    #         frame = cv2.imread(root + "/" + fdata)
    #         print('-> ' + root + "/" + fdata)
    #         faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    #         msg = ''
    #         if len(faces) > 0:
    #             faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    #             (x, y, w, h) = faces
    #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #             temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #             roi = temp[y:y + h, x:x + w]
    #             roi = cv2.resize(roi, (48, 48))
    #             roi = roi.astype("float") / 255.0
    #             roi = img_to_array(roi)
    #             roi = np.expand_dims(roi, axis=0)
    #             preds = image_sentiment_model.predict(roi)[0]
    #             emotion_probability = np.max(preds)
    #             label = EMOTIONS[preds.argmax()]
    #             msg = "Sentiment detected as : " + label
    #             text.insert(END, msg)
    #             main.update_idletasks()

    #             img_height, img_width = frame.shape[:2]
    #             cv2.putText(frame, msg, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #             cv2.imshow(fdata, frame)
    #             # cv2.imshow()
    #             # messagebox.showinfo(fdata, "Sentiment predicted from Facial expression as : " + label)
    #             if cv2.waitKey(10) & 0xFF == ord('q'):
    #                 break
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imwrite("sentimentImages/" + user + "-" + policy + ".jpg", img)
    # messagebox.showinfo("Your facial expression image accepted for reviews", "Your facial expression image accepted for reviews")

    # BELOW LINE DELETES THE SELECTED IMAGE FROM THE SENTIMENT_IMAGES
    # os.remove("sentimentImages/" + user + "-" + policy + ".jpg")
    # messagebox.showinfo("File has been removed", "File has been removed")


# we're not using this func, I've used this logic in other implementation
def photoSentiment():
    filename = 'sentimentImages'
    for root, dirs, files in os.walk(filename):
        for fdata in files:
            frame = cv2.imread(root + "/" + fdata)
            faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            msg = ''
            if len(faces) > 0:
                faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (x, y, w, h) = faces
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                temp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi = temp[y:y + h, x:x + w]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                preds = image_sentiment_model.predict(roi)[0]
                emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]
                msg = "Sentiment detected as : " + label
                img_height, img_width = frame.shape[:2]
                cv2.putText(frame, msg, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow(fdata, frame)
                messagebox.showinfo(fdata, "Sentiment predicted from Facial expression as : " + label)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def consolidatedResult():
    result = []

    try:
        # to check whether text_sentiment_model is been generated
        text_sentiment_model

        # 1. parsing images names in sentimentImages
        pathToSentimentImages = 'sentimentImages'
        fileNames = os.listdir(pathToSentimentImages)
        for fileName in fileNames:
            fileName = fileName[:-4]  # trim off   .jpg
            name, policy, sentiment = fileName.split('-')

            if policy not in result:
                result.extend([policy, 0, 0])  # [poilicy, +ve, -ve feedbacks]

            index = result.index(policy)
            if sentiment in POSITIVE_EMOTIONS:
                result[index + 1] += 1
            else:  # Negative
                result[index + 2] += 1

        # 2. read options.txt file and parse them
        pathToOpinions = 'Peoples_Opinion/opinion.txt'
        with open(pathToOpinions, "r") as file:
            # eg: Akash#It is a good policy#Pension
            for line in file:
                line = line.strip('\n')
                line = line.strip()
                arr = line.split("#")
                text_processed = stem(arr[1])
                X = [text_processed]
                sentiment = text_sentiment_model.predict(X)

                policy = arr[2]
                if policy not in result:
                    result.extend([policy, 0, 0])  # [poilicy, +ve, -ve feedbacks]

                index = result.index(policy)
                if sentiment[0] == 0:
                    # predicts = "Negative"
                    result[index + 2] += 1
                if sentiment[0] == 1:
                    # predicts = "Positive"
                    result[index + 1] += 1

        # 3. output the result
        consolidated_output = 'Consolidated Results:\n'
        for index in range(0, len(result) - 1, 3):
            poilcy = result[index]
            positive_feedback = result[index + 1]
            negative_feedback = result[index + 2]

            overall_result = ''
            if positive_feedback >= negative_feedback:
                overall_result = POSITIVE_RESULT
            else:
                overall_result = NEGATIVE_RESULT

            consolidated_output += f'''
Policy: {poilcy}
Positive Feedback: {positive_feedback}
Negative Feedback: {negative_feedback}
Overall Result: {overall_result}

'''
        text.delete('1.0', END)
        text.insert(END, consolidated_output)
    except NameError:
        text.delete('1.0', END)
        text.insert(END, "Please Generate the Deep Learning Models first.")

# ------------ UI

font = ('times', 16, 'bold')
# title = Label(main, text='Automating E-Government Services With Artificial Intelligence', anchor=W, justify=CENTER)
title = Label(main, text='Analysing Sentimental Reviews of E-Government Services using Deep Learning', anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=0)

# Generate Hand Written Digits Recognition Deep Learning Model
font1 = ('times', 14, 'bold')
digitButton = Button(main, text="Generate Hand Written Digits Recognition Deep Learning Model", command=digitModel)
digitButton.place(x=50, y=100)
digitButton.config(font=font1)

# Generate Text & Image Based Sentiment Detection Deep Learning Model
sentimentButton = Button(main, text="Generate Text & Image Based Sentiment Detection Deep Learning Model", command=sentimentModel)
sentimentButton.place(x=50, y=150)
sentimentButton.config(font=font1)

# Upload Image & Recognize Digit
recognizeButton = Button(main, text="Upload Image & Recognize Digit", command=digitRecognize)
recognizeButton.place(x=50, y=200)
recognizeButton.config(font=font1)

# Write Your Opinion About Government Policies
opinionButton = Button(main, text="Write Your Opinion About Government Policies", command=takeOpinion)
opinionButton.place(x=50, y=250)
opinionButton.config(font=font1)

# View Peoples Sentiments From Opinions
viewButton = Button(main, text="View Peoples Sentiments From Opinions", command=viewSentiment)
viewButton.place(x=50, y=300)
viewButton.config(font=font1)

# Upload & Detect Your Face Expression Photo About Government Policies
photoButton = Button(main, text="Upload & Detect Your Face Expression Photo About Government Policies", command=uploadPhotoAndDetectSentiment)
photoButton.place(x=50, y=350)
photoButton.config(font=font1)


# photosentimentButton = Button(main, text="Detect Sentiments From Face Expression Photo", command=photoSentiment)
photosentimentButton = Button(main, text="See Consolidated Result", command=consolidatedResult)
photosentimentButton.place(x=50, y=400)
photosentimentButton.config(font=font1)


font1 = ('times', 12, 'bold')

# Create Text Box
text = Text(main, height=17, width=75)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=700, y=120)
text.config(font=font1)


displayLabel = Label(main, text='Display:', anchor=W, justify=CENTER)
displayLabel.config(bg='grey', fg='green')
displayLabel.config(font=font)
# displayLabel.config(height=1, width=8)
displayLabel.place(x=700, y=90)


main.config(bg='grey')
main.mainloop()
