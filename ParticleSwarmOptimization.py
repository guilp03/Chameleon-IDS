import random
import RandomForest as rf
import XGBoost as gb
import dataset
import numpy as np
#random.seed(42)
#np.random.seed(42)

class Particle:
  def __init__(self,number ,incial_position, funct):
      self.velocity = [0] * 43 if funct == "rf" else [0] * 44
      self.position = incial_position
      self.personal_best = incial_position
      self.pb_val = 0
      self.pos_val = 0
      self.number = number
import pandas as pd
from particle_schema import Particle
from dataset import split_train_test
      
def Search_Space(funct: str, n_features: int):
    """ Função para definir os valores iniciais da partícula de acordo com o método de ensemble escolhido 
        
        Args:
            funct (str): rf para random forest ou gb para XGboost
            n_features (int): número de features de acordo com o dataset escolhido
        
        Returns:
            initial_position (array): configuração inicial da partícula da forma [test_size, ... features ..., hiperparametros de ensemble]
    """
    column_choice = [0, 1]
    initial_position = []
    initial_position.append(random.uniform(0.1, 0.4)) # Train-test-split
    for i in range(n_features):
        item = random.randint(0, 1)
        initial_position.append(item)
    if funct == "gb":
        initial_position.append(random.randint(100, 400))
        initial_position.append(random.uniform(0.05, 0.3))
    if funct == "rf":
        initial_position.append(random.randint(50, 600))
        print(initial_position)
    return initial_position


def Evaluate_fitness(funct: str, particle: Particle, ColumnsName: list[str], df: pd.DataFrame, y: pd.DataFrame, i: int, n_features: int | None =None):
    """ Função para calcular o f1_score de uma partícula

        Args:
            funct (str): rf para random forest ou gb para XGboost
            particle (Particle): partícula que se quer o f1_score
            ColumnsName (list[str]): lista de colunas do dataset das quais uma partícula pode selecionar
            df (pd.DataFrame): base de dados
            y (pd.DataFrame): labels
            i (int): índice da partícula na população
            n_features (int, optional): número de features a serem selecionadas

        Returns:
            f1_score (float) = f1_score da partícula escolhida
    
    """
    # Selecionando as colunas da partícula
    particle_choices = dataset.particle_choices(particle.position, ColumnsName, n_features)
    x_train, y_train, x_val, y_val = split_train_test(df, ColumnsName, y, particle.position[0])
    x_train_selected = x_train[particle_choices]
    x_val_selected = x_val[particle_choices]
    if funct == "gb":
    # Selecionar as colunas apropriadas
        gb_model = gb.GradientBoost(x_train_selected, y_train, particle.position[-2], particle.position[-1])
        accuracy_gb, f1_gb, precision_gb, recall_gb = gb.get_metrics(gb_model, x_val_selected, y_val)
        f1_gb = round(f1_gb, 4)
        feat_number= len(dataset.particle_choices(particle.position,ColumnsName, n_features=len(ColumnsName)))
        print(feat_number)
        fitness = round(0.8 * f1_gb + 0.2 * (1- (feat_number / n_features)), 4)
        print(particle.position)
        print(i,'accuracy:', accuracy_gb,'f1_score:',f1_gb, 'precision:', precision_gb, 'recall:', recall_gb, 'Features:', feat_number, "fitness:", fitness)
        return  fitness
    else:
        # Selecionar as colunas apropriadas
        rf_model = rf.RandomForest(n_features, x_train_selected, y_train,particle.position[-1])
        accuracy_rf, f1_rf, precision_rf, recall_rf = rf.get_metrics(rf_model, x_val_selected, y_val)
        print(particle.position)
        f1_rf = round(f1_rf, 4)

        print(i,'accuracy:', accuracy_rf,'f1_score:',f1_rf, 'precision:', precision_rf, 'recall:', recall_rf)
        return f1_rf
    
def checkvelocity(globalbest: list, particle: Particle, inertia: float, c1: int, c2: int):
    """ Calcula o array de velocidade de uma data partícula segundo o algorítmo PSO
        
        Args:
            globalbest (list): Melhor configuração encontrada para a partícula até então
            particle (Particle): partícula em questão
            inertia (float): hiperparâmetro da equação do PSO
            c1 (int): hiperparâmetro da equção do PSO
            c2 (int): hiperparâmetro da equção do PSO
    
        Returns:
            velocity (array): array do tamanho de particles.position que contém as velocidades de cada índice da partícula.
    """
    inertia_array = np.array([inertia])
    
    velocity =(list((particle.velocity * inertia_array) + c1 * random.random() * (np.array(particle.personal_best) - np.array(particle.position)) + c2 * random.random() * (np.array(globalbest) - np.array(particle.position))))
    #print(velocity)
    return velocity

def update_particle(particle: Particle, funct: str, n_features: int):
    """ Atualiza o valor de position de uma partícula

        Args:
            particle (Particle): partícula em questão
            funct (str): rf para RandomForest ou gb para XGBoost
            n_features (int): número de features do dataset

        Returns:
            position_array (list): novo position da partícula já com os valores tratados
    """
    for i in range(len(particle.position)):
        particle.position[i] += particle.velocity[i]
        
    particle.position = inteiro(particle, funct, n_features)

    return particle.position

def inteiro(particle: Particle, funct: str, n_features: int):
    """ Função para clipar cada valor de configuração em particle.position no intervalo desejado

        Args:
            particle (Particle): partícula em questão
            funct (str): rf para RandomForest ou gb para XGBoost
            n_features (int): número de features do dataset
        
        Returns:
            position_array (list): nova configuração de position com os valores clipados
    
    if funct == "rf":
        particle.position[-1] = np.clip(particle.position[-1], 50, 1000)
        particle.position[-1] = int(particle.position[-1])
    if funct == "gb":
        particle.position[-2] = np.clip(particle.position[-2], 50, 1000)
        particle.position[-2] =  int(particle.position[-2])
        particle.position[-1] = np.clip(particle.position[-1], 0.1, 0.3)
    """
    particle.position[0] = np.clip(particle.position[0], 0.1, 0.4)
    if funct == "rf":
        particle.position[-1] = int(np.clip(particle.position[-1], 50, 1000))
    if funct == "gb":
        particle.position[-2] = int(np.clip(particle.position[-2], 100, 400))
        particle.position[-1] = np.clip(particle.position[-1], 0.05, 0.3)
        particle.position[-1] = round(particle.position[-1], 4)
        
    for m in range(1, n_features + 1):
        if particle.position[m]<0.5:
            particle.position[m]=0
        if particle.position[m] >= 0.5:
            particle.position[m]=1
            
    return particle.position

def update_pb(particle: Particle):  
    """ Função para atualizar o personal_best de uma partícula 

        Args:
            particle (Particle): partícula em questão
        
        Returns:
            particle (Particle): mesma partícula agora com o valor de pb_val e personal_best modificado.
    
    """
    if particle.pb_val < particle.pos_val:
        particle.pb_val = particle.pos_val
        particle.personal_best = particle.position
        
    return particle

def check_globalbest(swarm, globalbest_val, globalbest):
    for particle in swarm:
        if particle.pb_val > globalbest_val:
            globalbest_val = particle.pb_val
            globalbest = particle.personal_best
    return globalbest_val, globalbest

#['id', 'duration', 'protocol_type', 'service', 'src_bytes', 'dst_bytes', 'land', 'hot', 'num_failed_logins', 'num_compromised', 'num_file_creations', 'is_host_login', 'srv_count', 'srv_rerror_rate', 'diff_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_rerror_rate']