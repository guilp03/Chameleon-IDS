import random
import RandomForest as rf
import XGBoost as gb
import dataset
import numpy as np
import pandas as pd
from particle_schema import Particle
      
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
        item = random.choice(tuple(column_choice))
        initial_position.append(item)
    if funct == "gb":
        initial_position.append(random.randint(50, 1000)) # Número de estágios de boosting
        initial_position.append(random.uniform(0.1, 0.3)) # Learning rate
    if funct.funct == "rf":
        initial_position.append(random.randint(50, 600)) # Número de árvores

    return initial_position

def Evaluate_fitness(funct: str, particle: Particle, ColumnsName: list[str], x_train: pd.DataFrame, y_train: pd.DataFrame | pd.Series, x_val: pd.DataFrame, y_val: pd.DataFrame | pd.Series, x_test: pd.DataFrame, y_test: pd.DataFrame | pd.Series, i: int, n_features: int | None =None):
    """ Função para calcular o f1_score de uma partícula

        Args:
            funct (str): rf para random forest ou gb para XGboost
            particle (Particle): partícula que se quer o f1_score
            ColumnsName (list[str]): lista de colunas do dataset das quais uma partícula pode selecionar
            x_train (pd.DataFrame): DataFrame de features para treinamento
            y_train (pd.Series | pd.DataFrame): Series ou DataFrame de target para treinamento
            x_val (pd.DataFrame): DataFrame de features para validação
            y_val (pd.Series | pd.DataFrame): Series ou DataFrame de target para validação
            x_test (pd.DataFrame): DataFrame de features para teste
            y_test (pd.Series | pd.DataFrame): Series ou DataFrame de target para teste
            i (int): índice da partícula na população
            n_features (int, optional): número de features a serem selecionadas

        Returns:
            f1_score (float) = f1_score da partícula escolhida
    
    """
    # Selecionando as colunas da partícula
    particle_choices = dataset.particle_choices(particle, ColumnsName, n_features)
    if funct == "gb":
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
    
def checkvelocity(globalbest: list, particle: Particle, inertia: float):
    """ Calcula o array de velocidade de uma data partícula segundo o algorítmo PSO
        
        Args:
            globalbest (list): Melhor configuração encontrada para a partícula até então
            particle (Particle): partícula em questão
            inertia (float): hiperparâmetro da equação do PSO
    
        Returns:
            velocity (array): array do tamanho de particles.position que contém as velocidades de cada índice da partícula.
    """
    inertia_array = np.array([inertia])
    
    velocity =(list((particle.position * inertia_array) + 2 * random.random() * (np.array(particle.personal_best) - np.array(particle.position)) + 2 * random.random() * (np.array(globalbest) - np.array(particle.position))))
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
    
    """
    particle.position[0] = np.clip(particle.position[0], 0.1, 0.4)
    if funct.funct == "rf":
        particle.potition[-1] = int(np.clip(particle.position[-1], 50, 1000))
    if funct.funct == "gb":
        particle.position[-2] = int(np.clip(particle.position[-2], 50, 1000))
        particle.potition[-1] = np.clip(particle.position[-1], 0.1, 0.3)
        
    for m in range(1, n_features + 1):
        if particle.position[m]>0.5:
            particle.position[m]=1
        else:
            particle.position[m]=0
            
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