import random
import RandomForest as rf
import XGBoost as gb
import dataset
import numpy as np

class Particle:
  def __init__(self,number ,incial_position):
      self.velocity = []
      self.position = incial_position
      self.personal_best = incial_position
      self.pb_val = 0
      self.pos_val = 0
      self.number = number
      
def Search_Space(funct):
    column_choice = [0, 1]
    inicial_position = []
    for i in range(42):
        item = random.choice(tuple(column_choice))
        inicial_position.append(item)
    if funct == "gb":
        inicial_position.append(random.randint(50, 1000))
        inicial_position.append(random.uniform(0.1, 0.3))
    if funct.funct == "rf":
        inicial_position.append(random.randint(50, 600))
    return inicial_position

def Evaluate_fitness(funct,particle, ColumnsName, x_train, y_train, x_val, y_val, x_test, y_test, i, n_features=None):
    if funct == "gb":
    # Selecionar as colunas apropriadas
        particle_choices = dataset.particle_choices(particle, ColumnsName)
        x_train_selected = x_train[particle_choices]
        x_val_selected = x_val[particle_choices]
        x_test_selected = x_test[particle_choices]
        gb_model = gb.GradientBoost(x_train_selected, y_train, particle.position[-2], particle.position[-1])
        accuracy_gb, f1_gb, precision_gb, recall_gb = gb.get_metrics(gb_model, x_val_selected, y_val)
        print(i,'accuracy:', accuracy_gb,'f1_score:',f1_gb, 'precision:', precision_gb, 'recall:', recall_gb)
        return f1_gb
    else:
        # Selecionar as colunas apropriadas
        x_train_selected = x_train[particle_choices]
        x_val_selected = x_val[particle_choices]
        x_test_selected = x_test[particle_choices]
        rf_model = rf.RandomForest(42, x_train_selected, y_train,particle.position[2])
        accuracy_rf, f1_rf, precision_rf, recall_rf = rf.get_metrics(rf_model, x_val_selected, y_val)
        print(i,'accuracy:', accuracy_rf,'f1_score:',f1_rf, 'precision:', precision_rf, 'recall:', recall_rf)
        return f1_rf
    
def checkvelocity(globalbest, particle, inertia):
    inertia_array = np.array([inertia])
    
    velocity =(list((particle.position * inertia_array) + 2 * random.random() * (np.array(particle.personal_best[j]) - np.array(particle.position)) + 2 * random.random() * (np.array(globalbest) - np.array(particle.position))))
    #print(velocity)
    return velocity

def update_particle(particle,funct):
    
    for i in range(len(particle.position)):
        particle.position[i] += particle.velocity[i]
        
    particle.position = inteiro(particle, funct)

    return particle.position

def inteiro(particle, funct):
    
    if funct.funct == "rf":
        particle.potition[-1] = np.clip(particle.position[-1], 50, 1000)
        particle.position[-1] = int(particle.position[-1])
    if funct.funct == "gb":
        particle.position[-2] = np.clip(particle.position[-2], 50, 1000)
        particle.position[-2] =  int(particle.position[-2])
        particle.potition[-1] = np.clip(particle.position[-1], 0.1, 0.3)
        
    for m in range(42):
        if particle.position[m]>0.5:
            particle.position[m]=1
        else:
            particle.position[m]=0
            
    return particle.position

def update_pb(particle):  
    if particle.pb_val < particle.pos_val:
        particle.pb_val = particle.pos_val
        particle.personal_best = particle.position
        
    return particle