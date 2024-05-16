import math
def euclid_distance(N, D, swarm, particle):
    """ Retorna a distância euclidiana de uma partícula a todas as outras partículas  
    """
    sum_qrt_d = 0
    for i in range(N):
        particle_aux = swarm[i]
        sum_d = 0
        if swarm[i] != particle:
            for j in range(D):
                sum_d += (particle_aux.position[j] - particle.position[j])**2
            sum_qrt_d += math.sqrt(sum_d)
        
    di = 1/(N-1) * sum_qrt_d
    
    return di

def evo_factor(dg, dmin, dmax):
    
    f = (dg - dmin) / (dmax - dmin)
    
    return f

def inertia_function(f):
    """ Retorna o valor atual da inércia
    """
    inertia = 1/ 1+1.5*math.e**(-2.6 * f)
    
    return inertia

def membership(f, state):
    """ Retorna o grau de pertencimento do evolutionary factor f ao estado passado como input
    """
    if state == "s1":
        if 0 <= f <= 0.4:
            membership = 0
        elif 0.4 < f <= 0.6:
            membership = 5 * f - 2
        elif 0.6 < f <= 0.7:
            membership = 1
        elif 0.7 < f <= 0.8:
            membership =-10 * f + 8
        elif 0.8 < f <= 1:
            membership = 0
    elif state == "s2":
        if 0 <= f <= 0.2:
            membership = 0
        elif 0.2 < f <= 0.3:
            membership = 10 * f - 2
        elif 0.3 < f <= 0.4:
            membership = 1
        elif 0.4 < f <= 0.6:
            membership = -5 * f + 3
        elif 0.6 < f <= 1:
            membership = 0
    elif state == "s3":
        if 0 <= f <= 0.1:
            membership = 1
        elif 0.1 < f <= 0.3:
            membership = -5 * f + 1.5
        elif 0.3 < f <= 1:
            membership = 0
    elif state == "s4":
        if 0 <= f <= 0.7:
            membership = 0
        elif 0.7 < f <= 0.9:
            membership = 5 * f - 3.5
        elif 0.9 < f <= 1:
            membership = 1
    return membership

def defuzzification(current_state, f):
    """ Tomada de decisões de troca de estados a partir do evolutionary factor e o estado atual"""
    membership = float
    if 0 <= f <= 0.2:
        new_state = "s3"
    elif 0.3 < f <= 0.4:
        new_state = "s2"
    elif 0.6 < f <= 0.7:
        new_state = "s1"
    elif 0.8 < f <= 1:
        new_state = "s4"
    elif 0.2 < f <= 0.3:
        if current_state == "s1" or current_state == "s2":
            new_state = "s2"
        elif current_state == "s3" or current_state == "s4":
            if membership(f, "s2") > membership(f, "s3"):
                new_state = "s2"
            else:
                new_state = "s3"
    elif 0.4 < f <= 0.6:
        if current_state == "s4" or current_state == "s1":
            new_state = "s1"
        elif current_state == "s2" or current_state == "s3":
            if membership(f, "s1") > membership(f, "s2"):
                new_state = "s1"
            else:
                new_state = "s2"
    elif 0.7 < f <= 0.8:
        if current_state == "s3" or current_state == "s4":
            new_state = "s4"
        elif current_state == "s2" or current_state == "s1":
            if membership(f, "s4") > membership(f, "s1"):
                new_state = "s4"
            else:
                new_state = "s1"
    return new_state
    
    
    