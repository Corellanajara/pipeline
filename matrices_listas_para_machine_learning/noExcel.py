import pandas as pd
import numpy as np
import sys
#import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc

from pandas.plotting import scatter_matrix
import math
import csv as lector
import os


def modelos():
    nombres = []
    #abro el resultado tratado que genero el script de abajo para luego pasarlo a python y poder usarlo
    documento = str(sys.argv[1])

    with open(documento, "rb") as f:
        data = list(lector.reader(f))
    i = 0
    for row in data:
        if i == 0:
            nombres = row
            i = i + 1
    # de los resultados tratados me interesa obtener los nombres
    # y generar un resultado para python que no tenga los nombres porque los paso por parametro


    ############  Propuesta para despues, pasar estas cosas de with open a una funcion que lo haga mas inteligente ##########
    with open("csv/resultadoParaPython.csv", "wb") as f:
        writer = lector.writer(f)
        i = 0
        for row in data:
            if i !=  0:
                if len(row)>0:
                    writer.writerow(row)
            i = i +1



    dataset = pd.read_csv("csv/resultadoParaPython.csv",names = nombres)
    print(dataset.describe())
    print(dataset.groupby('clase').size())
    # Split-out validation dataset
    array = dataset.values
    #obtengo los x que son todos los elementos menos el id y la clase
    X = array[:,1:len(nombres)-1]
    #clases o labels
    Y = array[:,len(nombres)-1]
    #ids
    Id = array[:0:1]

    validation_size = 0.20
    seed = 7
    scoring = 'accuracy'
    models = []

    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('Arbol', DecisionTreeRegressor()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(C=2 , kernel="rbf", gamma=0.009)))
    # Para evaluar todos los modelos los pasare en un ciclo for
    results = []
    names = []
    ListaPredicciones = []
    Confusiones = []
    for name, model in models:

        predicciones = []

        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        i = 0
        for train, test in kfold.split(X) :

            i = i +1
            #print ("train: " ,train," test :",test)
            X_train , X_test = X[train] , X[test]
            Y_train , Y_test = Y[train] , Y[test]
            modelo = model.fit( X_train,Y_train)
            pred = modelo.predict(X_test)
            predicciones.append(pred)




        pred = []

        for value in predicciones:
            for p in value:
                pred.append(p)
        ListaPredicciones.append(pred)

        print ("classifier : ", name)
        print("Puntaje precision :")
        print accuracy_score(Y, pred)
        print("Matriz de confusion :")
        print confusion_matrix(Y, pred)
        print "###################"
        cm = confusion_matrix(Y, pred)
        VP =  cm[0][0]
        FN = cm[0][1]
        FP = cm[1][0]
        VN = cm[1][1]
        #print VP
        #print FN
        #print FP
        #print VN
        Matriz = [VP,FN,FP,VN,name]
        Confusiones.append(Matriz)
        print(" Reporte clasificacion:")
        print classification_report(Y, pred)



    datas = []
    for prediccion in ListaPredicciones:
        fpr,tpr, _ = metrics.roc_curve(Y,prediccion)
        skplt.metrics.roc_curve(Y,prediccion)
        auc = metrics.roc_auc_score(Y,prediccion)
	print "Curva roc : "+str(auc)
        plt.plot(fpr,tpr,label="resultado , auc "+str(auc))
        plt.legend(loc=4)
        plt.show()
    for i in range(len(Confusiones)):
        print value
        vp = float(Confusiones[i][0])
        fn  = float(Confusiones[i][1])
        fp  = float(Confusiones[i][2])
        vn = float(Confusiones[i][3])
        modelo = str(Confusiones[i][4])

        prediccion = ListaPredicciones[i]
        fpr,tpr, _ = metrics.roc_curve(Y,prediccion)
        skplt.metrics.roc_curve(Y,prediccion)
        auc = metrics.roc_auc_score(Y,prediccion)
        n = "Clasificador : "+modelo
        p  = "precision : "+str(precision(vp,vn,fp,fn))
        s =  "sensibilidad : "+str(sensibilidad(vp,fn))
        e = "especificidad : "+str(especificidad(vp,fn))
        vpp = "valor Predictivo Positivo : "+str(valorPredictivoPositivo(vp,fp))
        vpn =  "Valor predictivo Negativo : "+str(valorPredictivoNegativo(vn,fn))
        m =  "MCC :"+str(mcc(vp,vn,fp,fn))

        #crear un csv
        # clasificado : {nombre}
        # auc : {valor}
        #todos : { todos }
        data = [n,p,s,e,vpp,vpn,m]
        datas.append(data)
    resume = pd.DataFrame(datas)
    resume.to_csv("resumen.csv")


def precision(vp,vn,fp,fn):
    return (vp+vn)/(vp+vn+fp+fn) * 100

def sensibilidad(vp,fn):
    return (vp)/(vp+fn)* 100

def especificidad(vn,fp):
    return (vn)/(vn+fp)* 100

def valorPredictivoPositivo(vp,fp):
    return (vp)/(vp+fp)* 100

def valorPredictivoNegativo(vn,fn):
    return (vn)/(vn+fn)* 100

def mcc(vp,vn,fp,fn):
    return ( (vp*vn)-(fp*fn) ) / ( math.sqrt((vp+fn)*(vp+fp)*(vn+fp)*(vn+fn) ) )* 100

def toCsv(content):
    data = False
    header = ""
    newContent = []
    for line in content:
        if not data:
            #un arff tiene esta estructura , entonces luego de analizarla la
            #dividi en partes para poder generar csv
            if "@attribute" in line:
                attri = line.split()
                columnName = attri[attri.index("@attribute")+1]
                header = header + columnName + ","
            elif "@data" in line:
                data = True
                header = header[:-1]
                header += '\n'
                newContent.append(header)
        else:
            newContent.append(line)
    return newContent



# ejecuto weka, paso la direccion de java , llego a los filtros supervisados y en seleccionador de atributo escojo best first y el evaluador para trabajar un archivo csv y convertirlo en un arff porque no permite otra salida
#documento = str(sys.argv[1])
#comando = "java -cp /usr/share/java/weka.jar weka.filters.supervised.attribute.AttributeSelection -E weka.attributeSelection.CfsSubsetEval -S weka.attributeSelection.BestFirst -i csv/"
#salida = "-o arff/resultadoTratado.arff"
#dir = comando + documento + salida
#os.system(dir)
#busco todos los archivos arff para pasarlos a csv
#files = [arff for arff in os.listdir('arff/.') if arff.endswith(".arff")]


# convierto el arrf y lo paso a csv porque no halle una libreria que lo hiciera por mi
#for file in files:
    #with open("arff/"+file , "r") as inFile:
    #    content = inFile.readlines()
    #    name,ext = os.path.splitext(inFile.name)
    #    a,b = name.split("/")
    #    new = toCsv(content)
    #    # genero el csv con los datos
    #    with open("csv/"+b+".csv", "w") as outFile:
    #        outFile.writelines(new)

# ejecuto los modelos para ver la diferencia con los modelos pasados por weka
#otro(masasPromedio)
# modelos es el que ejecuta los modelos luego de pasarlos por best first de weka
modelos()
