import numpy as np
import time
from statistics import median, mean
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from scipy import linalg

def steepest_descent_armijo(f, grad_f, x0, c1=1e-4, rho=0.5, tol=1e-6, k_max=150000):
    """Algoritmo de Descenso de Máximo Gradiente con line-search de Armijo."""
    x = np.copy(x0)
    hist_f = []
    hist_alpha = []
    hist_norm_g = []
    hist_backtracks = []

    start_time = time.time()
    k = 0
    stop_r = "max_iter"
    
    while k < k_max:
        fx = f(x)
        gx = grad_f(x)
        ng = np.linalg.norm(gx, 2)

        hist_f.append(fx)
        hist_norm_g.append(ng)
 
        if ng <= tol:
            stop_r = "||∇f|| ≈ 0"
            break

        pk = -gx
        alpha = 1.0
        n_backtracks = 0

        while f(x + alpha * pk) > fx + c1 * alpha * np.dot(gx, pk):
            alpha *= rho
            n_backtracks += 1
            if alpha < 1e-16:
                stop_r = "alpha_min"
                break

        if stop_r == "alpha_min":
            break

        x = x + alpha * pk
        hist_alpha.append(alpha)
        hist_backtracks.append(n_backtracks)
        
        k += 1

    total_time = time.time() - start_time
    if hist_alpha:
        alpha_1_pc = 100 * sum(1 for a in hist_alpha if a >= 1.0 - 1e-12) / len(hist_alpha)
        med_alpha = median(hist_alpha)
        avg_backtracks = mean(hist_backtracks)
    else:
        alpha_1_pc, med_alpha, avg_backtracks = 0.0, 0.0, 0.0

    return x, k, total_time, hist_f, hist_norm_g, hist_alpha, stop_r, (alpha_1_pc, med_alpha, avg_backtracks)


def rosenbrock(x):
    """Generador del valor de la función de Rosenbrock en base a un vector inicial."""
    n = len(x)
    suma = 0.0
    for i in range(n - 1):
        suma += 100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2
    return suma


def grad_rosenbrock(x):
    """Calcula el gradiente de la función de Rosenbrock."""
    n = len(x)
    grad = np.zeros(n)
    for i in range(n - 1):
        grad[i] += -400 * x[i] * (x[i + 1] - x[i]**2) - 2 * (1 - x[i])
        grad[i + 1] += 200 * (x[i + 1] - x[i]**2)
    return grad


def flatten(Z):
    """Aplana una matriz bidimensional en un vector unidimensional."""
    return Z.flatten()


def unflatten(z_vec):
    """
    Restaura la dimensionalidad de un vector aplanado a una matriz N_samples x P_dim.
    Asume que N_samples y P_dim existen en el entorno global.
    """
    return z_vec.reshape((N_samples, P_dim))


def stress_mds(z_vec, N_samples, D_target):
    """Calcula la pérdida para MDS (Stress)."""
    Z = unflatten(z_vec)
    loss = 0.0
    for i in range(N_samples):
        for j in range(i + 1, N_samples):
            dist_z = np.linalg.norm(Z[i, :] - Z[j, :])
            loss += (D_target[i, j] - dist_z)**2
    return 0.5 * loss


def grad_stress_mds(z_vec, N_samples, P_dim, D_target):
    """Calcula el gradiente analítico de la función Stress respecto a las posiciones Z."""
    Z = unflatten(z_vec)
    G = np.zeros((N_samples, P_dim))
    eps = 1e-12

    for i in range(N_samples):
        for j in range(N_samples):
            if i == j: 
                continue
            
            diff = Z[i, :] - Z[j, :]
            dist_z = np.linalg.norm(diff)
            grad_factor = (1 - D_target[i, j] / (dist_z + eps))
            G[i, :] += grad_factor * diff
            
    return flatten(G)


def distances_matrix(X):
    """Genera una matriz de distancias euclidianas por pares para un conjunto de datos X."""
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(X[i, :] - X[j, :])
    return D


def logistic_loss(z, X, y):
    """Calcula la log-likelihood negativa f(z) para la regresión logística."""
    n = len(y)
    u = X @ z
    loss = 0.0
    for i in range(n):
        if u[i] > 0:
            loss += (u[i] + np.log1p(np.exp(-u[i]))) - y[i] * u[i]
        else:
            loss += np.log1p(np.exp(u[i])) - y[i] * u[i]
    return loss


def logistic_grad_hess(z, X, y):
    """Calcula el gradiente y el Hessiano de f(z) para regresión logística."""
    u = X @ z
    pi = 1.0 / (1.0 + np.exp(-u))
    g = X.T @ (pi - y)
    W = pi * (1.0 - pi)
    B = (X.T * W[:, None]) @ X
    return g, B


def dogleg_trust_region(f, grad_hess_f, z0, delta_max=10.0, delta_k=1.0, eta=1e-4, tol=1e-6, k_max=1000):
    """Método de región de confianza implementando trayectoria Dogleg."""
    z = np.copy(z0)
    hist_f = []
    hist_norm_g = []
    hist_delta = []
    start_time = time.time()
    k = 0

    while k < k_max:
        fz = f(z)
        gk, Bk = grad_hess_f(z)
        norm_gk = np.linalg.norm(gk)

        hist_f.append(fz)
        hist_norm_g.append(norm_gk)
        hist_delta.append(delta_k)

        if norm_gk < tol:
            return z, k, time.time() - start_time, hist_f, hist_norm_g, hist_delta, "Converged"

        pk = compute_dogleg_step(gk, Bk, delta_k)
        actual_red = fz - f(z + pk)
        predicted_red = -(np.dot(gk, pk) + 0.5 * np.dot(pk, Bk @ pk))
        
        # Evitar división por cero
        rho_k = actual_red / predicted_red if predicted_red != 0 else 0

        if rho_k < 0.25:
            delta_k = 0.25 * delta_k
        else:
            if rho_k > 0.75 and np.isclose(np.linalg.norm(pk), delta_k):
                delta_k = min(2.0 * delta_k, delta_max)

        if rho_k > eta:
            z = z + pk
            
        k += 1

    return z, k, time.time() - start_time, hist_f, hist_norm_g, hist_delta, "Max iterations"


def compute_dogleg_step(g, B, delta):
    """Calcula el paso pk siguiendo la trayectoria entre el paso de Cauchy y el paso de Newton."""
    lam = 1e-6
    I = np.eye(B.shape[0])
    
    try:
        pb = -np.linalg.solve(B + lam * I, g)
    except np.linalg.LinAlgError:
        pb = -np.linalg.lstsq(B + lam * I, g, rcond=None)[0]
        
    norm_pb = np.linalg.norm(pb)

    if norm_pb <= delta:
        return pb

    alpha_c = np.dot(g, g) / np.dot(g, B @ g)
    pc = -alpha_c * g
    norm_pc = np.linalg.norm(pc)

    if norm_pc >= delta:
        return (delta / norm_pc) * pc

    diff = pb - pc
    a = np.dot(diff, diff)
    b = 2 * np.dot(pc, diff)
    c = np.dot(pc, pc) - delta**2
    
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        tau = max(0, -b / (2*a))
    else:
        tau = (-b + np.sqrt(discriminant)) / (2 * a)

    return pc + tau * diff


def load_mnist_direct():
    """Carga el conjunto de datos MNIST, filtra dígitos 0 y 1, y prepara la matriz de diseño."""
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

    def prepare_mnist(X_raw, y_raw):
        mask = (y_raw == 0) | (y_raw == 1)
        X_filtered = X_raw[mask]
        X_flat = X_filtered.reshape(X_filtered.shape[0], 784)
        y_filtered = y_raw[mask].astype(float)
        n_samples = X_flat.shape[0]
        X_tilde = np.hstack((X_flat, np.ones((n_samples, 1))))
        return X_tilde, y_filtered

    X_train, y_train = prepare_mnist(X_train_raw, y_train_raw)
    X_test, y_test = prepare_mnist(X_test_raw, y_test_raw)

    return X_train, y_train, X_test, y_test


def classification_error(z, X_test, y_test):
    """Calcula el error de clasificación promedio para el modelo de regresión logística."""
    u = X_test @ z
    probs = 1.0 / (1.0 + np.exp(-u))
    predictions = probs > 0.5
    return np.mean(np.abs(predictions - y_test))


def conjugate_gradient(A, b, max_iter=1000, tol=1e-9):
    """Algoritmo iterativo de Gradiente Conjugado para resolver sistemas de ecuaciones lineales."""
    hist_norm_rk = []
    N = b.shape[0]

    k = 0
    xk = np.zeros(N)
    rk = A @ xk - b
    pk = -rk
    norm_rk = np.linalg.norm(rk)
    hist_norm_rk.append(norm_rk)

    while norm_rk > tol and k < max_iter:
        Apk = A @ pk
        numerador = np.dot(rk, rk)
        denominador = np.dot(pk, Apk)
        
        alphak = numerador / denominador
        xk = xk + alphak * pk  
        rk = rk + alphak * Apk
        betak = np.dot(rk, rk) / numerador
        pk = -rk + betak * pk
        
        k += 1
        norm_rk = np.linalg.norm(rk) 
        hist_norm_rk.append(norm_rk)

    return xk, hist_norm_rk, k


def conjugate_gradient_precond(A, b, max_iter=1000, tol=1e-9):
    """Algoritmo de Gradiente Conjugado Precondicionado utilizando el precondicionador de Jacobi."""
    hist_norm_rk = []
    N = b.shape[0]

    k = 0
    xk = np.zeros(N)
    M = np.diag(A)
    rk = A @ xk - b
    yk = rk / M
    pk = -yk
    norm_rk = np.linalg.norm(rk)
    hist_norm_rk.append(norm_rk)

    while norm_rk > tol and k < max_iter:
        Apk = A @ pk
        numerador = np.dot(rk, yk)
        denominador = np.dot(pk, Apk)
        
        alphak = numerador / denominador
        xk = xk + alphak * pk  
        rk = rk + alphak * Apk
        yk = rk / M
        betak = np.dot(rk, yk) / numerador
        pk = -yk + betak * pk
        
        k += 1
        norm_rk = np.linalg.norm(rk) 
        hist_norm_rk.append(norm_rk)

    return xk, hist_norm_rk, k


def forwards_derivative_img(U, x, y, dim):
    """Calcula la derivada hacia adelante de una imagen en un pixel y dimensión específicos."""
    M, N = U.shape
    if dim == 1:
        return U[x+1, y] - U[x, y] if x < M - 1 else 0.0
    else:
        return U[x, y+1] - U[x, y] if y < N - 1 else 0.0


def backward_derivative_img(V, x, y, dim):
    """Calcula la derivada hacia atrás de un campo/matriz en un pixel y dimensión específicos."""
    if dim == 1:
        return V[x, y] - V[x-1, y] if x > 0 else 0.0
    else:       
        return V[x, y] - V[x, y-1] if y > 0 else 0.0


def gradient_approximation_img(U, x, y):
    """Aproxima el gradiente bidimensional de una imagen en un pixel usando diferencias hacia adelante."""
    dx = forwards_derivative_img(U, x, y, 1)
    dy = forwards_derivative_img(U, x, y, 2)
    return np.array([dx, dy])


def divergence_approximation_img(V_x, V_y, x, y):
    """Aproxima la divergencia de un campo vectorial bidimensional usando diferencias hacia atrás."""
    dx = backward_derivative_img(V_x, x, y, 1)
    dy = backward_derivative_img(V_y, x, y, 2)
    return dx + dy


def denoising(u0_matrix, total_iters):
    """Aplica un algoritmo de regularización de variación total (TV) para la reducción de ruido en imágenes."""
    u0 = np.array(u0_matrix, dtype=float)
    if np.max(u0) <= 1.0:
        u0 = u0 * 255.0
        
    M, N = u0.shape
    u = np.copy(u0)

    alpha = 0.0001
    gamma = 40.0
    eta = 0.0001

    hist_norm_grad_u = []
    dx = np.zeros((M, N))
    dy = np.zeros((M, N))
    norm_grad_u = np.zeros((M, N))
    div_V = np.zeros((M, N))
    
    hist_norm_grad_u.append(np.linalg.norm(norm_grad_u))

    for k in range(1, total_iters + 1):
        dx.fill(0.0)
        dy.fill(0.0)
        
        dx[:-1, :] = u[1:, :] - u[:-1, :]
        dy[:, :-1] = u[:, 1:] - u[:, :-1]

        norm_grad_u = np.sqrt(dx**2 + dy**2 + eta)
        hist_norm_grad_u.append(np.linalg.norm(norm_grad_u))
       
        dx = dx / norm_grad_u  
        dy = dy / norm_grad_u 

        div_V.fill(0.0)
        div_V[1:, :] += dx[1:, :] - dx[:-1, :]
        div_V[:, 1:] += dy[:, 1:] - dy[:, :-1]

        u = u - alpha * ((u - u0) - gamma * div_V)

        if k % 1000 == 0:
            print(f"Iteración {k} completada")
            u_visual = np.clip(u / 255.0, 0.0, 1.0) 
            plt.imshow(u_visual, cmap='gray')
            plt.axis('off')
            plt.show()

    return u, hist_norm_grad_u


def beale(x):
    """Evaluación de la función de Beale."""
    if len(x) != 2:
        raise ValueError("Array size must be 2")

    x1 = x[0]
    x2 = x[1]

    return (1.5 - x1 + x1 * x2)**2 + (2.25 - x1 + x1 * x2**2)**2 + (2.625 - x1 + x1 * x2**3)**2


def bealeGrad(x):
    """
    Evaluación del gradiente de beale en x   
    """
    n = len(x)
    if n != 2:
        raise ValueError("Array size must be 2")

    x1, x2 = x[0], x[1]
    g = np.zeros(2)

    f1 = 1.5 - x1 + x1*x2
    f2 = 2.25 - x1 + x1*x2**2
    f3 = 2.625 - x1 + x1*x2**3

    
    g[0] = 2*f1*(x2 - 1) + 2*f2*(x2**2 - 1) + 2*f3*(x2**3 - 1)
    
   
    g[1] = 2*f1*(x1) + 2*f2*(2*x1*x2) + 2*f3*(3*x1*x2**2)
    
    return g

def bealeHess(x):
    """
    Evaluación de la Hessiana de beale en x   
    """
    x_val = np.asarray(x).flatten()
    x1, x2 = x_val[0], x_val[1]
    
    f1 = 1.5 - x1 + x1 * x2
    f2 = 2.25 - x1 + x1 * x2**2
    f3 = 2.625 - x1 + x1 * x2**3

    H = np.zeros((2, 2))

    # d^2f / dx1^2
    H[0, 0] = 2 * (x2 - 1)**2 + 2 * (x2**2 - 1)**2 + 2 * (x2**3 - 1)**2

    # d^2f / dx2^2 

    term1 = 2 * x1**2
    term2 = 4 * x1 * f2 + 8 * (x1**2) * (x2**2)
    term3 = 12 * x1 * x2 * f3 + 18 * (x1**2) * (x2**4)
    H[1, 1] = term1 + term2 + term3

    # d^2f / dx1dx2
    cruz1 = 2 * f1 + 2 * x1 * (x2 - 1)
    cruz2 = 2 * (x2**2 - 1) * (2 * x1 * x2) + 4 * x2 * f2
    cruz3 = 2 * (x2**3 - 1) * (3 * x1 * x2**2) + 6 * x2**2 * f3
    H[0, 1] = H[1, 0] = cruz1 + cruz2 + cruz3

def himmelblau(x):
    """
    Evaluación de himmelblau en x   
    """
    
    x1, x2 = x[0], x[1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def himmelGrad(x):
    """
    Evaluación del gradiente de himmelblau en x   
    """

    x1, x2 = x[0], x[1]
    g = np.zeros(2)
    # df/dx1
    g[0] = 4 * x1 * (x1**2 + x2 - 11) + 2 * (x1 + x2**2 - 7)
    # df/dx2
    g[1] = 2 * (x1**2 + x2 - 11) + 4 * x2 * (x1 + x2**2 - 7)
    return g

def himmelHess(x):
    """
    Evaluación de la Hessiana de himmelblau en x   
    """
    x1, x2 = x[0], x[1]
    H = np.zeros((2, 2))
    # d2f/dx1^2
    H[0, 0] = 12 * x1**2 + 4 * x2 - 42
    # d2f/dx2^2
    H[1, 1] = 12 * x2**2 + 4 * x1 - 26
    # d2f/dx1dx2
    H[0, 1] = H[1, 0] = 4 * x1 + 4 * x2
    return H

def hartmann(x):
    """
    Evaluación de hartmann en x   
    """

    x = np.asarray(x)
    if x.ndim == 1:
        n = len(x)
    else:
        n = x.shape[1]
    if n != 6:
        raise ValueError("Array dimensions must be 6")
    
    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]])
    
    sum = 0
    for i in range(4):
        sum += alpha[i] * np.exp(-np.sum(A[i,:] * (x - P[i,:])**2)) 

    return -(2.58 + sum) / 1.94
    
def hartmannGrad(x):
    """
    Evaluación del gradiente de hartmann en x   
    """
    x = np.asarray(x)
    if x.ndim == 1:
        n = len(x)
    else:
        n = x.shape[1]

    if n != 6:
        raise ValueError("Array dimensions must be 6")
    
    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]])

    grad = np.zeros(6)


    #E_i
    E = np.zeros(4)
    for i in range(4):
        E[i] = np.exp(-np.sum(A[i,:] * (x - P[i,:])**2))

    for k in range(6):
        s = 0.0
        for i in range(4):
            s += alpha[i]*E[i]*A[i, k]* (x[k] - P[i,k])
        grad[k] = 2.0 / 1.94 * s

    return grad



def hartmanHess(x):
    """
    Evaluación de la Hessiana de hartmann en x   
    """
    x = np.asarray(x).flatten()
    if len(x) != 6:
        raise ValueError("Array dimensions must be 6")

    alpha = np.array([1.0, 1.2, 3.0, 3.2])

    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091, 381]])

    H = np.zeros((6,6))
    E = np.zeros(4)
    for i in range(4):
        E[i] = np.exp(-np.sum(A[i, :] * (x - P[i, :])**2))


    for k in range(6):
        for m in range(6):
            s = 0.0
            for i in range(4):
                delta_km = 1.0 if k == m else 0.0

                inn = delta_km - 2 * A[i, m] * (x[k] - P[i, k]) * (x[m] - P[i, m])
                s += alpha[i] * A[i,k] * E[i] * inn

        H[k,m] = 2.0 / 1.94 * s

    return H
        

def rosenbrockHessTri(x):
    """
    Evaluación de la Hessiana de Rosenbrock en x (Es una matriz tridiagonal simétrica entonces la almacenamos como una tupla de d, du, du)   
    """

    x = np.asarray(x).flatten()
    n = len(x)

    d = np.zeros(n)
    du = np.zeros(n - 1)

    for i in range(n - 1):
        d[i] += 1200 * x[i]**2 - 400 * x[i+1] + 2
        d[i+1] += 200
        du[i] = -400 * x[i]

    return (d, du, du)


def get_eigenvalue_extremes(matrix):
    """
    Calcula los autovalores de una matriz simétrica y devuelve el mayor y el menor.
    
    Args:
        matrix (np.array): Matriz simétrica.
        
    Returns:
        tuple: (lambda_min, lambda_max)
    """

    eigenvalues = np.linalg.eigvalsh(matrix)
    
    lambda_1 = np.min(eigenvalues)
    lambda_n = np.max(eigenvalues)
    
    return lambda_1, lambda_n

def check_matrix_definiteness(lambda_1, lambda_n):
    """
    Imprime los autovalores extremos y clasifica la matriz según su definición.
    
    Args:
        lambda_1 (float): Autovalor mínimo.
        lambda_n (float): Autovalor máximo.
    """
    print(f"Lowest eigenvalue (lambda_1): {lambda_1:.6e}")
    print(f"Highest eigenvalue (lambda_n): {lambda_n:.6e}")
    
    if lambda_1 > 0:
        print("Definite positive matrix")
    elif lambda_n < 0:
        print("Definite negative matrix")
    elif lambda_1 < 0 and lambda_n > 0:
        print("Indefinite matrix")
    else:
        #En caso de que alguno sea 0
        print("Semi-definite or Singular matrix")


def newtonDescent(f, grad, hess, x0, max_iter = 10000, alpha_0 = 1, c1 = 0.1, rho = 0.6, bt_max_iter = 100):
    """
    Función de optimización usando la dirección de Descenso de Newton

    Args:
        f       (function): Función a optimizar.
        grad    (function): Gradiente de la función.
        hess    (function): Hessiana de la función.
        x0      (np.array): Punto inicial

    Returns:
        x_k:        (np.array)  : Valor final de la función
        g_k:        (np.array)  : Valor final del gradiente
        m:          (int)       : Cantidad de veces que se uso pk = -gk
        res:        (int)       : 1 en caso de convergencia
        trajectory  (list)      : Lista para n<=2 para verificar trayectoria 
    
    """

    m = 0
    k = 0
    x_k = np.asarray(x0, dtype=float).flatten()
    n = len(x_k)
    tol = np.sqrt(n * np.finfo(float).eps)
    res = 0
    trajectory = [x_k.copy()] if n == 2 else []

    print(f"f(x0) = {f(x_k):.6e}")

    for k in range(max_iter):

        g_k = grad(x_k)
        norm_gk = linalg.norm(g_k)

        if norm_gk < tol:
            print(f"Iter: {k} | ||gk||: {norm_gk:.2e} | f(x): {f(x_k):.4e} | m: {m}")
            if n <= 6:
                print(f"\nx_k: {x_k}")
            else: #Print de los primeros 3 y utlimos 3
                print(f"\nx_k (first/last 3) : {x_k[:3]} ... {x_k[-3:]}")
            res = 1
            break

        B_k = hess(x_k)
        
        try:
            #Caso de Rosenrock donde la Hessiana es tridiagonal 
            if isinstance(B_k, tuple) and len(B_k) == 3:
                d, du, dl = B_k

                ab = np.zeros((2, n))
                ab[0, 1:] = du 
                ab[1, :] = d                 
                p_k = linalg.solveh_banded(ab, -g_k, lower=False)
            
            # Caso donde la hessiana es densa
            else:
                c, low = linalg.cho_factor(B_k)
                p_k = linalg.cho_solve((c, low), -g_k)

        except (linalg.LinAlgError, ValueError):
            p_k = -g_k
            m += 1

        #Backtracking
        alpha = alpha_0
        bt_iter = 0
        while bt_iter < bt_max_iter:
            if f(x_k + alpha * p_k) <= f(x_k) + c1 * alpha * np.dot(g_k, p_k):
                break
            alpha *= rho
            bt_iter += 1
            
        x_k = x_k + alpha * p_k
        
        if n == 2:
            trajectory.append(x_k.copy())

        if (k + 1) % 100 == 0 or k < 5: 
            print(f"Iter: {k} | ||gk||: {norm_gk:.2e} | f(x): {f(x_k):.4e} | m: {m}")
            # Print de todo el vector
            if n <= 6:
                print(f"\nx_k: {x_k}")
            else: #Print de los primeros 3 y utlimos 3
                print(f"\nx_k (first/last 3) : {x_k[:3]} ... {x_k[-3:]}")

    

    return x_k, g_k, k, m, res, trajectory


def modifiedBFGS(f, grad, x0, max_iter = 10000, alpha_0 = 1, c1 = 0.1, rho = 0.6, max_bt_iter = 100):
    
    x_k = np.asarray(x0, dtype=float).flatten()
    n = len(x_k)
    g_k = grad(x_k)
    I = np.identity(n)
    H_k = np.identity(n)
    res = 0
    tol = np.sqrt(n * np.finfo(float).eps)
    trajectory = [x_k.copy()] if n == 2 else []

    for k in range(max_iter):
        
        norm_gk = np.linalg.norm(g_k)
        if norm_gk < tol:
            print(f"Iter: {k} | ||gk||: {norm_gk:.2e} | f(x): {f(x_k):.4e}")
            if n <= 6:
                print(f"\nx_k: {x_k}")
            else: #Print de los primeros 3 y utlimos 3
                print(f"\nx_k (first/last 3) : {x_k[:3]} ... {x_k[-3:]}")
            res = 1
            break

        p_k = - (H_k @ g_k)
        pk_gk_dot = np.dot(p_k, g_k)

        if (pk_gk_dot > 0):

            gk_gk_dot = np.dot(g_k, g_k)
            lambda_1 = 10**(-5) + pk_gk_dot / gk_gk_dot
            H_k = H_k + lambda_1 * I
            p_k = p_k - lambda_1 * g_k

        #Bactracking
        alpha = alpha_0
        bt_iter = 0
        while bt_iter < max_bt_iter:
            if f(x_k + alpha * p_k) <= f(x_k) + c1 * alpha * np.dot(g_k, p_k):
                break
            alpha *= rho
            bt_iter += 1

        x_next = x_k + alpha * p_k
        g_next = grad(x_next)

        s_k = x_next - x_k
        y_k = g_next - g_k

        yk_sk_dot = np.dot(s_k, y_k)
        if (yk_sk_dot <= 0):

            yk_yk_dot = np.dot(y_k, y_k)
            lambda_2 = 10**(-5) - yk_sk_dot / yk_yk_dot            
            H_k = H_k + lambda_2 * I

        else:
            rho_k = 1.0 / yk_sk_dot
            A1 = I - rho_k * np.outer(s_k, y_k)
            A2 = I - rho_k * np.outer(y_k, s_k)
            H_k = A1 @ H_k @ A2 + (rho_k * np.outer(s_k, s_k))

        x_k = x_next
        g_k = g_next

        if n == 2:
            trajectory.append(x_k.copy())
        
        if (k + 1) % 100 == 0 or k < 5: 
            print(f"Iter: {k} | ||gk||: {norm_gk:.2e} | f(x): {f(x_k):.4e}")
            # Print de todo el vector
            if n <= 6:
                print(f"\nx_k: {x_k}")
            else: #Print de los primeros 3 y utlimos 3
                print(f"\nx_k (first/last 3) : {x_k[:3]} ... {x_k[-3:]}")
            
    return x_k, g_k, k, res, trajectory