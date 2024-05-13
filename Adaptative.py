import math
def euclid_distance(N, D, swarm, particle):
    sum_d = 0
    for i in range(N):
        particle_aux = swarm[i]
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
    inertia = 1/ 1+1.5*math.e**(-2.6 * f)
    
    return inertia

def f_value(f, case):
    if case == 0:
        if 0 <= f <= 0.4:
            f = 0
        elif 0.4 < f <= 0.6:
            f = 5 * f - 2
        elif 0.6 < f <= 0.7:
            f = 1
        elif 0.7 < f <= 0.8:
            f =-10 * f + 8
        elif 0.8 < f <= 1:
            f = 0
    elif case == 1:
        if 0 <= f <= 0.2:
            f = 0
        elif 0.2 < f <= 0.3:
            f = 10 * f - 2
        elif 0.3 < f <= 0.4:
            f = 1
        elif 0.4 < f <= 0.6:
            f = -5 * f + 3
        elif 0.6 < f <= 1:
            f = 0
    elif case == 2:
        if 0 <= f <= 0.1:
            f = 1
        elif 0.1 < f <= 0.3:
            f = -5 * f + 1.5
        elif 0.3 < f <= 1:
            f = 0
    elif case == 3:
        if 0 <= f <= 0.7:
            f = 0
        elif 0.7 < f <= 0.9:
            f = 5 * f - 3.5
        elif 0.9 < f <= 1:
            f = 1
    return f