import json
import pickle
import random


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import  Dense,Activation,Dropout
from tensorflow.python.keras.optimizers import  adam_v2
import numpy as np
import nltk
from nltk.stem import wordnet
import self
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer= wordnet.WordNetLemmatizer
intenciones= json.loads(open("intents.json").read())
palabras=[]
clases=[]
documents=[]
ignore_letters=[]

for intenciones in intenciones["intenciones"]:
  for patrones in intenciones["patrones"]:
   lista_palabras=nltk.word_tokenize(patrones)
   palabras.extend(lista_palabras)
   documents.append((lista_palabras, intenciones["tag"]))
   if intenciones["tag"] not in clases:
    clases.append(intenciones["tag"])

palabras=[lemmatizer.lemmatize(self,palabra) for palabra in palabras if palabra not in ignore_letters]
palabras=sorted(set(palabras))
clases=sorted(set(clases))
pickle.dump(palabras, open("palabras.pkl", "wb"))
pickle.dump(clases, open("clases.pkl", "wb"))

trainning=[]
output_empty=[0]* len(clases)
for document in documents:
 bow=[]
 patrones_palabras=document[0]
 patrones_palabras=[lemmatizer.lemmatize(self,palabra.lower()) for palabra in patrones_palabras]
 for palabra in palabras:
   if palabra in patrones_palabras:
    bow.append(1)
   else:
    bow.append(0)
   output_row= list(output_empty)
   output_row[clases.index(document[1])]=1
   trainning.append((bow,output_row))
random.shuffle((trainning))
trainning=np.array(trainning)
train_x=list(trainning[:,0])
train_y=list(trainning[:,1])

modelo=Sequential()
modelo.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
modelo.add(Dropout(0.5))
modelo.add(Dense(64, activation="relu"))
modelo.add(Dropout(0.5))
modelo.add(Dense(len(train_y[0]), activation="softmax"))
adam=adam_v2.Adam(lr=0.01, decay=1e-6)
modelo.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
hist=modelo.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)


modelo.save("chatbot_modelo.h5", hist)
print("Hecho.")
