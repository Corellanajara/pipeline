import pandas as pd
import numpy as np
#import scikitplot as skplt
from sklearn import model_selection
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from pandas.plotting import scatter_matrix
from scipy.stats import ttest_ind
from scipy import stats
import statistics as estadistica
import sys
import math
import csv as lector
import os


# pido la planilla excel mediante pandas
excel = pd.ExcelFile('planilla-ensayo.xlsx')
xFinal = []
yFinal = []
zFinal = []
masas = []
masasaPromedio = []

# deje todo esto comentado pero luego debe quitarse para usarse con datos del usuario

#cortemin = input("Ingrese el valor minimo para discriminar datos :  ")
#cortemax = input("Ingrese el valor maximo para discriminar datos :  ")

#while cortemax < cortemin:
#    print "El valor maximo debe ser mayor al minimo\n"
#    cortemin = input("Ingrese el valor minimo para discriminar datos :  ")
#    cortemax = input("Ingrese el valor maximo para discriminar datos :  ")
#error = input("Cual es el porcentaje de error para este documento?")
#Sderror = input("Cual es el valor de error de calibracion del equipo?")

cortemin = 5000;
cortemax = 14000;
error = 4;
Sderror = 4;
def generarPdf():
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    canvas = canvas.Canvas("form.pdf", pagesize=letter)
    canvas.setLineWidth(.3)
    canvas.setFont('Helvetica', 12)

    canvas.drawString(30,750,'CARTA DE PRUEBA')
    canvas.drawString(30,735,'RICARDOGEEK.COM')
    canvas.drawString(500,750,"27/10/2016")
    canvas.line(480,747,580,747)

    canvas.drawString(275,725,'ESTIMADO:')
    canvas.drawString(500,725,"<NOMBRE>")
    canvas.line(378,723,580,723)

    canvas.drawString(30,703,'ETIQUETA:')
    canvas.line(120,700,580,700)
    canvas.drawString(120,703,"<ASUNTO DE LA CARTA GENERICO>")

    canvas.save()
# funcion generar matriz recibe las hojas y las masas promedio y los estados
def generarMatriz(hojas,masasPromedio,IdEstado):
    x = len(hojas) + 1
    y = len(masasPromedio)+1
    porcentajeError = Sderror - (Sderror*0.2)
    #genero una matriz en cero de rango x y
    matriz = np.zeros((x,y))
    ids = []
    estados = []

    for id in IdEstado:
        result = id.split("-")
        ids.append(result[0])
        estados.append(result[1])

    for i in range(x ):
        veces = 0
        for j in range(y):
            if ( i == 0 ):
                if j != y - 1:
                    valor = masasPromedio[j]
                    #print valor
                    #matriz[i][j] = valor
                if j == 0 :
                    # a todos los primeros elementos le asigno el id
                    matriz[i+1][j] = float(ids[i])

                if ( j==(y-1) ):
                    # a todos los ultimos elementos le asigno el id
                    matriz[i+1][j] = float(estados[i])
            else :
                if ( j==0  ):
                    if i < len(ids):
                        matriz[i+1][j] = float(ids[i])
                    pass
                else :
                    # encontrar los valores de z
                    if i < x :
                        hoja = hojas[ i - 1 ]
                        tam = len(hoja)
                        veces = veces + 1
                        if j < tam:
                            valor = masasPromedio[j]
                            valormas = valor + porcentajeError
                            valormenos = valor - porcentajeError
                            index = 0
                            #encontramos los elementos x para buscar su valor en z
                            for elemento in hoja['x']:
                                if ( elemento >= valormenos and elemento <=valormas ):
                                    # si existe el valor dentro de los rangos establecidos se almacena
                                    matriz[i][j] = float(hoja['z'][index])
                                index = index + 1
                    if ( j==(y-1) ):
                        if i < len(ids):
                            matriz[i+1][j] = float(estados[i])

    Matriz = []
    i = 0
    for m in matriz:
        if  i == 0 :
            pass
        else :
            Matriz.append(m)
        i = i + 1


    resultado = pd.DataFrame(Matriz,columns=None)
    #guardo la matriz sin los nombres de las clases

    resultado.to_csv("csv/resultado.csv",index=False,index_label=False)
#    print matriz
    return resultado

#funcion de alineamiento
def alineamiento():
    SderrorLLenado =  Sderror - (Sderror*0.2)
    #ordeno de menor a mayor
    masas.sort()
    i = 0
    j = 0
    temp = []
    promedios = []
    siguienteSet = True
    valor = 0
    suma = 0

    while ( i < len(masas) ):
        #de el total del arreglo
        #pregunto por cada set de datos
        if(siguienteSet):
            #asigno el nuevo valor maximo y minimo

            temp = []
            valor = masas[i]
            valorMax = valor + SderrorLLenado
            valorMin = valor - SderrorLLenado
            siguienteSet = False
        if ( masas[i] <= valorMax ):
            #cuando es menor al valor maximo
            if  masas[i] >= valorMin :
                # y a la vez es mayor al valor  maximo

                suma = suma + masas[i]
                j = j + 1
        else:
            if(j!=0):
                #agrego un elemento a los promedios
                suma = suma / j
                promedios.append(suma)
            suma = 0
            j = 0
            siguienteSet = True
            #permito el paso al siguiente set
        i = i + 1
    ####################

    xFinal = np.around(np.array(promedios),3)
    #redondeo los datos

    return xFinal

def sumatoriaMasas(contenido):
    for elemento in contenido['x']:
        masas.append(elemento)

def normalizar(paso1):
    #aplicar la formula de normalizacion
    paso1 = np.round(paso1,decimals=3)
    valorMaximo = paso1['Intens.'].max()
    masa = []
    intensidad = []
    z = []
    for i in range(len(paso1['Intens.'])):
        porcentaje = (paso1['Intens.'][i]*100)/valorMaximo
        masa.append(paso1['masa'][i])
        intensidad.append(paso1['Intens.'][i])
        z.append(np.round(porcentaje,decimals=3))
    paso2 = pd.DataFrame({'x' : masa,'y':intensidad,'z':z})

    return paso2

def reduccionDimensionalidad(paso2):

    #aplico filtros de minimos y maximos para quitar los que no cumplan
    x ,y,z= []  , [] ,[]
    for i in range( len(paso2) ):
        if((cortemax > paso2['x'][i] and paso2['x'][i] > cortemin ) and paso2['z'][i] > error):
            x.append(paso2['x'][i])
            y.append(paso2['y'][i])
            z.append(paso2['z'][i])
    # x = masas  y = intensidad z = llenado
    paso3 = pd.DataFrame({'x' : x,'y':y,'z':z})

    return paso3

def otro(nombres):
    dataset = pd.read_csv("csv/resultado.csv",names = nombres)
    # print(dataset.describe())
    # print(dataset.groupby('clase').size())

    array = dataset.values

    X = array[:,1:len(nombres)-1]

    Y = array[:,len(nombres)-1]

    Id = array[:0:1]

    validation_size = 0.20
    seed = 7

    scoring = 'accuracy'
    models = []

    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))

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

            X_train , X_test = X[train] , X[test]
            Y_train , Y_test = Y[train] , Y[test]

            modelo = model.fit(X_train,Y_train)
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
        #skplt.metrics.roc_curve(Y,prediccion)
        auc = metrics.roc_auc_score(Y,prediccion)
	print "Curva roc : "+str(auc)
        plt.plot(fpr,tpr,label="resultado , auc "+str(auc))
        plt.legend(loc=4)
        plt.show()
    for i in range(len(Confusiones)):
        # print value
        vp = float(Confusiones[i][0])
        fn  = float(Confusiones[i][1])
        fp  = float(Confusiones[i][2])
        vn = float(Confusiones[i][3])
        modelo = str(Confusiones[i][4])

        prediccion = ListaPredicciones[i]
        fpr,tpr, _ = metrics.roc_curve(Y,prediccion)
        #skplt.metrics.roc_curve(Y,prediccion)
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

def modelos():
    nombres = []
    #abro el resultado tratado que genero el script de abajo para luego pasarlo a python y poder usarlo
    with open("csv/resultadoTratado.csv", "rb") as f:
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
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # Para evaluar todos los modelos los pasare en un ciclo for
    results = []
    names = []

    for name, model in models:

        predicciones = []

        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        i = 0
        for train, test in kfold.split(X) :
            i = i +1
            #print ("train: " ,train," test :",test)
            X_train , X_test = X[train] , X[test]
            Y_train , Y_test = Y[train] , Y[test]

            modelo = model.fit(X_train,Y_train)
            pred = modelo.predict(X_test)
            predicciones.append(pred)


        pred = []

        for value in predicciones:
            for p in value:
                pred.append(p)


        print ("classifier : ", name)
        print("Puntaje precision :")
        print accuracy_score(Y, pred)
        print("Matriz de confusion :")
        print confusion_matrix(Y, pred)
        print(" Reporte clasificacion:")
        print classification_report(Y, pred)


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


atributos = []
contador = 0 ;
data = {}
paso1 = pd.DataFrame(data)
paso2 = pd.DataFrame(data)
paso3 = pd.DataFrame(data)
pasos3 = []
hojas = []
X = []
Y = []
atributosX = []
atributosY = []
flag1 = 0
flag2 = 0
#clase1 =  pd.DataFrame([1, 1, 1, np.nan], index=['a', 'b', 'c', 'd'],columns=['clase1']))
for hoja in excel.sheet_names:
    #genero los archivos que usare
    sheet = pd.read_excel("planilla-ensayo.xlsx",sheet_name=hoja)
    sheet.to_csv('recursos/hojas/hoja'+str(hoja)+'.csv',index=False)

    csv =  pd.read_csv('recursos/hojas/hoja'+str(hoja)+'.csv')
    clase = hoja.split("-");
    elementosX = []
    elementosY = []
    indice = 0



    #leo masa e intensidad
    masa = csv['masa']
    intensidad = csv['Intens.']

    paso1 = pd.concat([masa.reset_index(drop=True), intensidad], axis=1)
    #normalizo
    paso2 = normalizar(paso1)
    #elimino datos innecesarios
    paso3 = reduccionDimensionalidad(paso2);
    #junto todos los pasos 3 que son los datos previo a la reduccion de dimensionalidad
    pasos3.append(paso3)
    paso3.to_csv("recursos/OK-"+str(hoja)+".csv",index=False)
    #tengo la cantidad de hojas para luego iterar en ellas
    hojas.append(hoja)
    sumatoriaMasas(paso3)


#print " este es X"
#print len(X)
#print " este es Y"
#print len(Y)
with open("csv/resultadoTratado.csv", "rb") as f:
    data = list(lector.reader(f))
i = 0
for row in data:
    if i == 0:
        nombres = row
        i = i + 1

Datos = pd.read_csv("csv/resultadoParaPython.csv",names = nombres)

# Split-out validation dataset
promedioX = []
promedioY = []
desviacionX = []
desviacionY = []
mannwhitneyu = []
wilcoxon = []
kruskal = []
testudend = []
X = []
Y = []
contador = 0
array = Datos.values
resultados = []
for i in range(len(nombres)-1):
    X = []
    Y = []
    for j in range(len(array)):
        if(array[j][len(nombres)-1]==0 or array[i][len(nombres)-1]== '0' ):
            X.append(array[j][i])
        else :
            Y.append(array[j][i])


    Test = [["wilcoxon",stats.wilcoxon(X,Y)],["testudent",stats.ttest_ind(X,Y)],["mannwhitneyu",stats.mannwhitneyu(X,Y)],["kruskal",stats.kruskal(X,Y)]]
#        print atributosX[i]
#        print "valor clase 1 : "+str(X[i])
#        print "valor clase 2 : "+str(Y[i])

    data_to_plot = [X,Y]
    promedioX.append(estadistica.mean(X))
    promedioY.append(estadistica.mean(Y))
    desviacionX.append(estadistica.pstdev(X))
    desviacionY.append(estadistica.pstdev(Y))

    for nombre , test in Test:
        print "datos"
        print "para hipotesis "+str(nombre) + " los datos son :"
        stat , p = test
        print "stat : "+str(stat)
        print "p : "+str(p)
        if (nombre == 'mannwhitneyu'):
            mannwhitneyu.append(p)
        if (nombre == 'kruskal'):
            kruskal.append(p)
        if (nombre == 'testudent'):
            testudend.append(p)
        if (nombre == 'wilcoxon'):
            wilcoxon.append(p)

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(data_to_plot)
    ax.set_title(nombres[i])
    ax.set_xlabel('Clases')
    ax.set_ylabel('Valores')
    ax.set_xticklabels(['Clase 1','Clase 2'])
    fig.savefig(str(nombres[i])+'.png', bbox_inches='tight')
    plt.show()
nombres.pop()
matrizPvalues = {'atributos':nombres,'p value mannwhitneyu':mannwhitneyu,'p value kruskal':kruskal,'p value t estudent':testudend,'mean class 1':promedioX,'mean class 2':promedioY, 'sd class 1':desviacionX,'sd class 2':desviacionY}
pValues = pd.DataFrame(data=matrizPvalues)
pValues.to_csv("csv/pValues.csv")
#obtengo los x que son todos los elementos menos el id y la clase


    #stat , p = ttest_ind(X[i],Y[i])
    #data = stat,p;


#print resultados
#stat,p = stats.mannwhitneyu(X[i],Y[i])
#Test = [stats.ttest_ind(X[i],Y[i]),stats.mannwhitneyu(X[i],Y[i]),stats.wilcoxon(X[i],Y[i]),stats.kruskal(X[i],Y[i])]
#Resultados = []
#for test in Test:
    #stat , p = test


#alineamiento multiple de las masas
masasPromedio = list(alineamiento())
#le inserto el atributo clase
masasPromedio.insert(len(masasPromedio) + 1,'clase')
labels = masasPromedio
#guardo los labels
#genero la matriz que a su vez, genera el archivo resultado.csv
final = generarMatriz(pasos3,masasPromedio,hojas)

masasPromedio.insert(0,'id')

#abro el archivo que resulta del ciclo anterior para obtener sus datos
with open("csv/resultado.csv", "rb") as f:
    data = list(lector.reader(f))

# genero dos archivos uno el resultado csv y el otro el resultadoWEKA

#resultado weka incluye los labels para que weka pueda tratarlos
with open("csv/resultadoWEKA.csv", "wb") as f:
    writer = lector.writer(f)
    i = 0
    for row in data:
        if i==0:
            writer.writerow(labels)
        if i != 0:
            writer.writerow(row)
        i = i +1

#este va sin labels porque se trabaja asi, se pasa los labels por separado
with open("csv/resultado.csv", "wb") as f:
    writer = lector.writer(f)
    i = 0
    for row in data:
        if i != 0:
            writer.writerow(row)
        i = i +1

# ejecuto weka, paso la direccion de java , llego a los filtros supervisados y en seleccionador de atributo escojo best first y el evaluador para trabajar un archivo csv y convertirlo en un arff porque no permite otra salida
os.system("java -cp /usr/share/java/weka.jar weka.filters.supervised.attribute.AttributeSelection -E weka.attributeSelection.CfsSubsetEval -S weka.attributeSelection.BestFirst -i csv/resultadoWEKA.csv -o arff/resultadoTratado.arff")
#busco todos los archivos arff para pasarlos a csv
files = [arff for arff in os.listdir('arff/.') if arff.endswith(".arff")]


# convierto el arrf y lo paso a csv porque no halle una libreria que lo hiciera por mi
for file in files:
    with open("arff/"+file , "r") as inFile:
        content = inFile.readlines()
        name,ext = os.path.splitext(inFile.name)
        a,b = name.split("/")
        new = toCsv(content)
        # genero el csv con los datos
        with open("csv/"+b+".csv", "w") as outFile:
            outFile.writelines(new)

# ejecuto los modelos para ver la diferencia con los modelos pasados por weka
otro(masasPromedio)
# modelos es el que ejecuta los modelos luego de pasarlos por best first de weka
modelos()
