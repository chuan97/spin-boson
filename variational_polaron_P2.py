import numpy as np

def f_deltar(deltar, J, delta, x, env):
    return delta * np.exp(-2 * np.sum(f_f(deltar, J, x, env) ** 2))

def f_f(deltar, J, x, env):
    E = f_E(deltar, J)
    num = E + J*np.cos(env.ks * x)
    return env.gs/env.ws * num/(num + (deltar ** 2)/env.ws)

def f_J(deltar, J, x, env):
    fs = f_f(deltar, J, x, env)
    return 2 * np.sum((fs*(2*env.gs - env.ws*fs) - env.gs**2/env.ws) * np.cos(env.ks * x))

def f_E(deltar, J):
    return np.sqrt(deltar**2 + J**2)

def variational_deltar(delta, x, env):
    deltar = 0.001
    J = 0.001
    
    while True:
        new_deltar = f_deltar(deltar, J, delta, x, env)
        new_J = f_J(deltar, J, x, env)
        change = max(abs(deltar - new_deltar), abs(J - new_J))

        deltar, J = new_deltar, new_J

        if change < 1e-9: #cuando la mejora es ya muy pequeña cierro el bucle
            break
    
    return deltar, J