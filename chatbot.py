import  random
import json
import pickle

import nltk
import numpy as np
from tensorflow.python.keras.models import load_model
from nltk.stem import wordnet

lemmatizer=wordnet.WordNetLemmatizer()
intenciones=json.loads(open("intents.json").read())
palabras=pickle.load(open("palabras.pkl", "rb"))
clases=pickle.load(open("clases.pkl", "rb"))
modelo= load_model("chatbot_modelo.model")

def limpiar_frase(frase):
    palabras_frase= nltk.wordpunct_tokenize(frase)
    palabras_frase=[lemmatizer.lemmatize(palabra) for palabra in palabras_frase]
    return palabras_frase
def bag_of_words(frase):
    palabras_frase=limpiar_frase(frase)
    bag=[0]*len(palabras)
    for p in palabras_frase:
        for i, palabra in enumerate(palabras):
            if palabra==p:
                bag[i]=1
    return np.array(bag)
def predcit_class(frase):
    bow=bag_of_words(frase)
    res=modelo.predict(np.array([bow]))[0]
    ERROR_THRESHOLD= 0.25
    resultado=[[i,r] for i, r in enumerate (res) if r> ERROR_THRESHOLD]
    resultado.sort(key=lambda x: x[1], reverse=True)
    return_lista=[]
    for r in resultado:
        return_lista.append({"intenciones": clases[r[0]], "probability": str(r[1])})
    return return_lista
def chatear(lista_intenciones, intenciones_json):
    tag= lista_intenciones[0]["intenciones"]
    l_intenciones= intenciones_json["intenciones"]
    for i in l_intenciones:
        if i["tag"] == tag:
            res=random.choice(i["respuestas"])
            break
    return res


print("Estas chateando! Di algo!")

while True:
    message= input("")
    inst= predcit_class(message)
    res=chatear(inst, intenciones)
    print (res)