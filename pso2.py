from sklearn import preprocessing
import dataset_supervised as ds
preprocessing.LabelEncoder()
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option("display.max_columns", None)

columnsName1=[0,1]
particles=[]
for i in range(10):
    part1=[]
    for i in range(56):
        item = random.choice(tuple(columnsName1))
        part1.append(item)
    particles.append(part1)
    
    
def data(particles1):
    particles2=[]
    for i in range(len(particles1)):
        if particles1[i]!=1:
                particles2.append(ds.columnsName[i])
    return particles2

pb=[]
def checkpersonalnest(particles):
    for i in range(len(particles)):
         pb.append(ds.accuracy_calc(data(particles[i])))
checkpersonalnest()

def checkvelocity(globalbest, particles):
    velocity=[]
    for j in range(len(particles)):
        velocity.append(list(0+1*(np.random.random(1)[0])*(np.array(particles[j])-np.array(particles[j]))+1*(np.random.random(1)[0])*(np.array(globalbest)-np.array(particles[j]))))
    #print(velocity)
    return velocity

def addingparticles(velocity, particles):
    particles2=[]
    for i in range(len(velocity)):
        nextchromo=[]
        for j in range(len(velocity[i])):
            nextchromo.append(particles[i][j]+velocity[i][j])
        particles2.append(nextchromo)
    return particles2

def normalize(particles2):
    for l in range(len(particles2)):
        for m in range(len(particles2[l])):
            if particles2[l][m]>0.5:
                particles2[l][m]=1
            else:
                particles2[l][m]=0
    return particles2

def checkpd(particles2, particles):
    personal=[]
    for i in range(len(particles2)):
        personal.append(ds.accuracy_calc(data(particles2[i])))
    for j in range(len(personal)):
        if(personal[j]>pb[j]):
            particles[j]=particles2[j]
            pb[j]=personal[j]
    return personal

max(pb)
ind = pb.index(max(pb))
globalbest=particles[ind]
for i in range(10):
    particles2=[]
    personal=[]
    velocity=checkvelocity(globalbest, particles)
    particles2=addingparticles(velocity, particles)
    particles2=normalize(particles2)
    personal=checkpd(particles2, particles)
    particles = particles2
    globalbest=[]
    max(pb)
    ind = pb.index(max(pb))
    globalbest=particles[ind]
                
    
max(pb)

ind = pb.index(max(pb))
globalbest=particles[ind]

print(data(globalbest))

#FALTA CHECAR A ATUALIZAÇÃO DE VALORES DE PARTICLES