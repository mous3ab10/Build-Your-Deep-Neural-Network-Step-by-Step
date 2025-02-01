import numpy as np
import copy
from utils import sigmoid, sigmoid_backward, relu, relu_backward

# ------------------------- Initialisation -------------------------
def initialize_parameters(n_x, n_h, n_y):
    """Initialisation pour un réseau à 2 couches."""
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def initialize_parameters_deep(layer_dims):
    """Initialisation pour un réseau profond (L couches)."""
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
    return parameters

# ------------------------- Propagation avant -------------------------
def linear_forward(A, W, b):
    """Étape linéaire de la propagation avant."""
    Z = W.dot(A) + b
    return Z, (A, W, b)

def linear_activation_forward(A_prev, W, b, activation):
    """Étape linéaire + activation (ReLU/Sigmoid)."""
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    return A, (linear_cache, activation_cache)

def L_model_forward(X, parameters):
    """Propagation avant complète pour un réseau L-couches."""
    caches = []
    A = X
    L = len(parameters) // 2
    
    # Couches ReLU (L-1 fois)
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters[f'W{l}'], parameters[f'b{l}'], "relu")
        caches.append(cache)
    
    # Dernière couche: Sigmoid
    AL, cache = linear_activation_forward(A, parameters[f'W{L}'], parameters[f'b{L}'], "sigmoid")
    caches.append(cache)
    return AL, caches

# ------------------------- Calcul du coût -------------------------
def compute_cost(AL, Y):
    """Calcule le coût d'entropie croisée."""
    m = Y.shape[1]
    cost = -(1/m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    return np.squeeze(cost)

# ------------------------- Propagation arrière -------------------------
def linear_backward(dZ, cache):
    """Étape linéaire de la rétropropagation."""
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m) * dZ.dot(A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T.dot(dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """Étape linéaire + activation inverse."""
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """Rétropropagation complète pour le réseau L-couches."""
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    
    # Initialisation
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Dernière couche (Sigmoid)
    current_cache = caches[L-1]
    grads[f'dA{L-1}'], grads[f'dW{L}'], grads[f'db{L}'] = linear_activation_backward(dAL, current_cache, "sigmoid")
    
    # Couches ReLU (de L-2 à 1)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads[f'dA{l+1}'], current_cache, "relu")
        grads[f'dA{l}'] = dA_prev_temp
        grads[f'dW{l+1}'] = dW_temp
        grads[f'db{l+1}'] = db_temp
        
    return grads

# ------------------------- Mise à jour des paramètres -------------------------
def update_parameters(params, grads, learning_rate):
    """Met à jour les paramètres avec la descente de gradient."""
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2
    
    for l in range(L):
        parameters[f'W{l+1}'] -= learning_rate * grads[f'dW{l+1}']
        parameters[f'b{l+1}'] -= learning_rate * grads[f'db{l+1}']
        
    return parameters