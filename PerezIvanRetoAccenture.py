#PRIMERO, IMPORTO TODAS LAS LIBRERÍAS NECESARIAS

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords

import string

import spacy
import spacy_spanish_lemmatizer ; nlp = spacy.load('es_core_news_sm') ; nlp.replace_pipe('lemmatizer', 'spanish_lemmatizer')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import xgboost
from xgboost import XGBClassifier

import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

#CARGO EL DATASET CON LAS FRASES
df_raw = pd.read_excel(r'frases.xlsx')


#FUNCIONES USADAS:

def crear_target(df_raw): #Nos pone un 1 si la frase fue generada por una IA y un 0 si la generó un humano
    df = df_raw.copy() #Para no cambiar el dataset original
    df.loc[df['Tipo texto'] == 'Humano', 'Tipo texto'] = 0 ; df.loc[df['Tipo texto'] == 'IA', 'Tipo texto'] = 1
    return df

def lema_stopwords(frase): #Esta función toma una frase y le hace la primera parte del preprocesado: Quita palabras sin significado (stopwords)...
                           #... y transforma todas las palabras a su lematizado (ver PDF para mayor comprensión)
    stop_words_esp = set(stopwords.words('spanish'))
    frase_nueva = ''
    for j in nlp(frase):  #Comprueba que el lematizado no sea una de las stopwords y si es así la añade a la frase nueva):
        if str(j.lemma_) not in stop_words_esp:
            frase_nueva += str(j.lemma_) + ' '
    frase_nueva = frase_nueva.translate(str.maketrans('', '', string.punctuation)) #Quita los puntos, comas, etc.
    return frase_nueva

def prep_df_final(df): #Termina el preprocesado del dataset transformando todas las frases y haciendo la vectorización:
    df2 = df.copy()
    for i in range(len(df2)):
        df2.loc[i, 'Frases'] = lema_stopwords(df2.loc[i, 'Frases'])
    
    vect = TfidfVectorizer() ; frases = list(df2['Frases']) #Ahora vectorizo cada una de las frases

    count_matrix = vect.fit_transform(frases)
    count_array = count_matrix.toarray()
    df3 = pd.DataFrame(data=count_array,columns = vect.get_feature_names())
    df_final = pd.concat([df3, df2['Tipo texto']], axis=1)
    
    return df_final

def separ_train_test(df_final, tamano_test): #Básicamente es train_test_split de sklearn, pero lo hago como función para editar test_size
    X = df_final.iloc[:,:-1] ; y = df_final.iloc[:,-1].astype(int)
    X_train, X_test , y_train, y_test = train_test_split(X, y , test_size = tamano_test)
    
    return X_train, X_test, y_train, y_test

def modelo_clasificacion(classifier, X_train, X_test, y_train, y_test): #Función que toma como input el modelo que queramos. Lo entrena y saca...
                                                                        #... los resultados obtenidos.
    classifier.fit(X_train,y_train) #Entrena al modelo
    
    pred = classifier.predict(X_test) 
    
    #Resultados que nos pueden interesar
    accuracy = accuracy_score(y_test,pred)
    print(f'The {classifier}  Accuracy  is {accuracy}' )
    tabla_resultados = classification_report(y_test, pred) ; print(tabla_resultados) #Muestra métricas como F1-score, recal...
        #Matriz de confusión
    matriz_confusion = confusion_matrix(y_test, pred)
    plt.figure() ; sns.heatmap(matriz_confusion, annot = True, cmap="BrBG", fmt = 'g')
    plt.xlabel('Valores predichos del target') ; plt.ylabel('Valores conocidos del target') ; plt.title(f'Confusión para modelo {classifier}')
    if accuracy >= 0.75: #Solo considero los modelos que tengan una precisión razonable, si es menor el modelo saca los rsultados pero no las pred
        return pred #Voy a usar esto para el modelo híbrido / democrático

    

#AHORA METEMOS LOS PRIMEROS MODELOS QUE QUEREMOS (la idea es mezclar todos para obtener un modelo híbrido):
    #Primero preparo todo siguiendo los pasos:
df = crear_target(df_raw) ; plt.figure() ; sns.countplot(data = df, x = 'Tipo texto') ; plt.title('Balance de las dos clases de frases')
df_final = prep_df_final(df) ; X_train, X_test, y_train, y_test = separ_train_test(df_final, 0.3) #Tamaño arbitrario test

    #Meto los primeros algoritmos que no requieren de preparación previa y que puedo hacer usando un bucle
classifiers = [GradientBoostingClassifier(),GaussianNB(),HistGradientBoostingClassifier(), 
               RandomForestClassifier(),LogisticRegression(),XGBClassifier()]
predicciones = []
for i in classifiers:
    predicciones.append(modelo_clasificacion(i, X_train, X_test, y_train, y_test))

    
    
    #Creo una red neuronal secuencial sencilla como siguiente modelo para ver si mejora lo anterior
red_neuronal = tf.keras.models.Sequential()
red_neuronal.add(tf.keras.layers.Dense(units=4, activation='relu')) #Capa input
#red_neuronal.add(tf.keras.layers.Dense(units=2, activation='relu')) #Capa interna
red_neuronal.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #Capa output. Función sigmoide como activación por la naturaleza del problema
red_neuronal.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
red_neuronal.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)
        #Aquí hay que tener en cuenta que los valores output de la red son probabilidades de que las frases sean 1. Tomo como umbral un 0.5
        # Esto significa que si P > 0.5, la red predice que la frase es un 1 (creada por IA) y viceversa.
keras_predict = red_neuronal.predict(X_test)
keras_predict = (keras_predict > 0.5)   # Umbral que comento
my_list = map(lambda x: x[0], keras_predict)
keras_predict = pd.Series(my_list) ; keras_predict = keras_predict.astype(int) #Lo que he explicado
        #Muestro mismo esquema de resultados que para los anteriores
print(f'The Keras NN  Accuracy  is {accuracy_score(y_test,keras_predict)}' )
tabla_resultados = classification_report(y_test, keras_predict) ; print(tabla_resultados) #Muestra métricas como F1-score, recal...
matriz_confusion = confusion_matrix(y_test, keras_predict) ; plt.figure() ; sns.heatmap(matriz_confusion, annot = True, cmap="BrBG", fmt = 'g')
plt.xlabel('Valores predichos del target') ; plt.ylabel('Valores conocidos del target') ; plt.title(f'Confusión para modelo Keras NN')

if accuracy_score(y_test,keras_predict) >= 0.75: #Mismo criterio de antes
    predicciones.append(keras_predict)
predicciones = [x for x in predicciones if x is not None] #Me quedo solo con las predicciones de los mejores.
                

                     
# AHORA MEZCLAMOS TODOS LOS RESULTADOS SIGUIENDO UN RAZONAMIENTO DEMOCRÁTICO:
prediccion_hibrido = []
for i in range(len(predicciones[0])): #Todos tienen la misma longitud:
    n_zeros, n_ones = 0,0
    for j in range(len(predicciones)):
        if predicciones[j][i] == 0:
            n_zeros += 1
        else:
            n_ones += 1
    if n_zeros > n_ones:
        prediccion_hibrido.append(0)
    else:
        prediccion_hibrido.append(1)

print(f'The Hybrid Model Accuracy  is {accuracy_score(y_test,prediccion_hibrido)}' )
tabla_resultados = classification_report(y_test, prediccion_hibrido) ; print(tabla_resultados) #Muestra métricas como F1-score, recal...
matriz_confusion = confusion_matrix(y_test, prediccion_hibrido) ; plt.figure() ; sns.heatmap(matriz_confusion, annot = True, cmap="BrBG", fmt = 'g')
plt.xlabel('Valores predichos del target') ; plt.ylabel('Valores conocidos del target') ; plt.title(f'Confusión para modelo híbrido')
plt.show()
