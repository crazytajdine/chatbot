import json
import nltk 
import tensorflow
import numpy as np
model = tensorflow.keras.models.load_model("chatbot.h5")
stemmer = nltk.stem.LancasterStemmer()
with open("modeluseddata.json","r") as wrdsjs : 
    js = json.load(wrdsjs)
    words = js[0]
    y = js[1]
# type quit to quit
text = ""
while(text != "quit"):
    text = input("you:")
    txt = nltk.word_tokenize(text)
    txt = list(set([stemmer.stem(word.lower()) for word in txt]))
    xtest = np.array([[1 if wrd in txt else 0 for wrd in words]])
    with open("dataset.json","r") as dataset:
        print("AI :",y[np.argmax(model.predict(xtest,verbose=None))])