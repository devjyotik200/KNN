import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Social_Network_Ads.csv')

x=dataset.iloc[:300,2].values
y=dataset.iloc[:300,3].values
z=dataset.iloc[:300,4].values

f=dataset.iloc[300:,4].values

q=dataset.iloc[:300,[2,3]].values
q1=dataset.iloc[300:,[2,3]].values

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x5=sc_x.fit_transform(q)
x6=sc_x.fit_transform(q1)

def distance(x1,y1,x2,y2):
    return (((x1-x2)**2)+((y1-y2)**2))**(0.5)

ar=[]

def knn(x,z,a):  #X IS THE ALREADY GIVEN COORDINATES. Z IS THE ALREADY GIVEN POINTS AND a IS THE TESTING VALUES
    for j in range(0,100):
        cz=0 
        cx=0
        dict={}
        for i in range(0,300):
            dict[i]=distance(x[i][0],x[i][1],a[j][0],a[j][1])
        sort = sorted(dict.items(), key=lambda x: x[1])
        for i in range(0,5):
            if z[sort[i][0]]==0:
                cz+=1
            elif z[sort[i][0]]==1:
                cx+=1
        print(cz,cx)
        if cx>cz:
            ar.append(1)
            print("the person's probably gonna buy the car")
        else:
            ar.append(0)
            print("the person's probably not gonna buy the car")
knn(x5,z,x6)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ar,f)
