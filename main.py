import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas
import pandas as pd
from mpl_toolkits import mplot3d
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from mpl_toolkits import mplot3d

# Datensatz laden
iris = pd.read_csv(
    "test2.csv",
    sep=";",
    decimal=",",
    #header=0,
    #thousands='.',
    #quoting=csv.QUOTE_NONNUMERIC,
    engine='python'
    #skiprows=1 #Überspringe die erste Zeile der Datei
    )

#Konvertiere Object zu String
iris['Vorhersage'] = pandas.Series(iris['Vorhersage'], dtype="string")

#Gebe alle Datentypen der in iris enthaltenen Elemente aus
print(iris.dtypes)

#Erstelle Instanzen der Figur und der Axen zur späteren Plot-Bearbeitung
fig,ax = plt.subplots(1)

#ToDo: Definieren der Farbe nach Vorhersagewert

# Dataframe zu Numpy Array
#dataframe = iris.DataFrame({'newgazetimestamp', 'Winkelgeschwindigkeit'})
number_column = iris.loc[:,'Winkelgeschwindigkeit']
numbers = number_column.values

iris.as_matrix(columns=iris.columns[])
#iris[iris('newgazetimestamp'), iris('Winkelgeschwindigkeit')].to_numpy()
groups = iris.groupby('Vorhersage')

#Daten plotten
for name, group in groups:
    plt.plot(group['newgazetimestamp'], ['Winkelgeschwindigkeit'], marker="o", label=name)
    print(groups.get_group('Fixation'))
    print(groups.get_group('Sakkade'))
    

#Deaktiviere Wertbeschriftung an den Axen
#ax.set_yticklabels([])
#ax.set_xticklabels([])

#Begrenzt angezeigten Wertebereich
#plt.xlim(-40, 40)
plt.ylim(0, 1000)

#Diagrammbeschriftung
plt.title("")
plt.xlabel("Zeit x")
plt.ylabel("Winkelgeschwindigkeit y")

#Plot & Legende anzeigen
plt.legend()
print("here")
plt.show()

#for name, group in groups:
#    plt.plot(group['Winkelgeschwindigkeit'], ['Winkel'], marker="o", label=name)
#plt.xlabel("Winkelgeschwindigkeit")
#plt.ylabel("Winkel")
#plt.xlim(0, 1000)
#plt.legend()
#plt.show()



#ToDo: Überlegen, ob eine zufällige Wahl der Trainings- & Testdaten möglich ist!
#Trainings-und Testdaten erzeugen
#x_train,x_test,y_train,y_test=train_test_split(
#    np.array(iris['newgazetimestamp']),
#    np.array(iris['Winkelgeschwindigkeit']),
#    test_size=0.3)



