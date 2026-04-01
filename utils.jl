using LinearAlgebra
using Statistics
using Pickle
using GZip
using PyCall
using MLDatasets
using DelimitedFiles

"""
steepest_descent_armijo(f, grad_f, x0; c1, rho, tol, k_max)

Algoritmo de Descenso de Máximo Gradiente con line-search de Armijo.

# Parámetros:
- `f`: Función objetivo
- `grad_f`: Función gradiente de f en el punto x.
- `x0`: Punto inicial
- `c1`: Constante de Armijo
- `rho`: Factor de reducción 
- `tol`: Tolerancia 
- `k_max`: Máximo de iteraciones

# Retorno:
- `x`: Punto final 
- `k`: Número de iteraciones
- `total_time`: Tiempo de ejecución
- `hist_f`, `hist_norm_g`, `hist_alpha`: Historiales de la función, gradiente y pasos.
- `stop_r`: Razón de finalización ("||∇f|| ≈ 0", "max_iter", "alpha_min").
- `stats`: Tupla con estadísticas (porcentaje de pasos unidad, mediana de alpha, promedio backtracks).
"""

function steepest_descent_armijo(f, grad_f, x0; c1 = 1e-4, rho = 0.5, tol = 1e-6, k_max = 150000)
    x = copy(x0)

    #Vectores par almacenar el historial de lso valores
    hist_f = Float64[]
    hist_alpha = Float64[]
    hist_norm_g = Float64[]
    hist_backtracks = Int[]

    start_time = time()
    k = 0
    stop_r = "max_iter"
    
    while k < k_max
        fx = f(x)
        gx = grad_f(x)
        ng = norm(gx, 2)

        # Almacenamos lso valores
        push!(hist_f, fx)
        push!(hist_norm_g, ng)

        # Criterio de parada
        if ng <= tol
            stop_r = "||∇f|| ≈ 0"
            break
        end

        pk = -gx
        alpha = 1.0
        n_backtracks = 0

        #COndicion de armijo
        while f(x + alpha * pk) > fx + c1 * alpha * dot(gx, pk)
            alpha *= rho
            n_backtracks += 1
            # Parada para α muy pequeño
            if alpha < 1e-16
                stop_r = "alpha_min"
                break
            end
        end

        #Parada de emergencia en caso de que ya no se pueda escojer otro α
        if (stop_r) == "alpha_min"; break; end

        x .+= alpha * pk
        push!(hist_alpha, alpha)
        push!(hist_backtracks, n_backtracks)
        
        k += 1
    end

    total_time = time() - start_time
    #Estadisticas de α, porcentaje, mediana y promedio de backtracks por iteracion
    if !isempty(hist_alpha)
        alpha_1_pc = 100 * count(a -> a >= 1.0 - 1e-12, hist_alpha) / length(hist_alpha)
        med_alpha = median(hist_alpha)
        avg_backtracks = mean(hist_backtracks)
    else
        alpha_1_pc, med_alpha, avg_backtracks = 0.0, 0.0, 0.0
    end

    return x, k, total_time, hist_f, hist_norm_g, hist_alpha, stop_r, (alpha_1_pc, med_alpha, avg_backtracks)
end

""" 
 function rosenbrock(x)

 Generador del vector de rosenbrock en base a un vector inicial

 # Parámetros:
 - `x`: Vector de dimension N 

 # Retorno:
- `suma`: Evaluación de la funcion

"""
function rosenbrock(x)
    n = length(x)
    suma = 0.0
    for i in 1:(n-1)
        suma += 100 * (x[i + 1] - x[i]^2)^2 + (1 - x[i])^2
    end
    return suma
 end 

 function grad_rosenbrock(x)
    n = length(x)
    g = zeros(n)
    for i in 1:(n - 1)
        g[i]        += -400 * x[i] * (x[i + 1] - x[i]^2) - 2 * (1 - x[i])
        g[i + 1]    += 200 * (x[i + 1] - x[i] ^ 2)
    end
    return g
end

 """
 Funciones auxiliares para el MDS 
 
 """
flatten(Z) = vec(Z)
unflatten(z_vec) = reshape(z_vec, N_samples, P_dim)

"""
    stress_mds(z_vec, N_samples, D_target)

Calcula la loss para MDS.
Representa la discrepancia entre las distancias en el espacio original y el proyectado.

# Parámetros:
- `z_vec`: Vector aplanado de coordenadas en baja dimensión (N*P).
- `N_samples`: Número total de puntos
- `D_target`: Matriz de distancias euclidianas
"""
 function stress_mds(z_vec, N_samples, D_target) 
    Z = unflatten(z_vec)
    loss = 0.0
    for i in 1:N_samples
        for j in (i + 1):N_samples
            dist_z = norm(Z[i, :] - @view Z[j, :]) #Distancia en 2D
            loss += (D_target[i, j] - dist_z)^2 #Error al cuadrado
        end
    end
    return 0.5 * loss
end 

"""
    grad_stress_mds(z_vec, N_samples, P_dim, D_target)

Calcula el gradiente analítico de la función Stress respecto a las posiciones Z.

# Parámetros:
- `P_dim`: Dimensión del espacio de proyección (típicamente 2).
- `eps`: Pequeña constante para evitar división por cero si dos puntos coinciden.


"""

function grad_stress_mds(z_vec, N_samples, P_dim, D_target);
    Z = unflatten(z_vec)
    G = zeros(N_samples, P_dim)
    eps = 1e-12

    for i in 1:N_samples
        for j in 1:N_samples
            if i == j; continue; end
            
            diff = Z[i, :] - Z[j, :]
            dist_z = norm(diff)
            grad_factor = (1 - D_target[i, j] / (dist_z + eps))
            G[i, :] += grad_factor * diff
        end
    end
    return flatten(G)
end

"""
    distances_matrix(X)

Genera una matriz de distancias euclidianas por pares para un conjunto de datos X.
"""
function distances_matrix(X);
    n = size(X, 1)
    D = zeros(n, n)
    for i in 1:n, j in 1:n
        D[i, j] = norm(X[i, :] - X[j, :])
    end
    return D
end

"""
    logistic_loss(z, X_tilde, y)

Calcula la log-likelihood negativa f(z)
- X_tilde Matriz de diseño 
- y: Vector de etiqueta {0, 1}
"""

function logistic_loss(z, X, y)
    # Muestras
    n = length(y)

    # Esto es igual a x_i^T + z
    u = X * z

    loss = 0.0
    for i in 1:n
        if u[i] > 0
            loss += (u[i] + log1p(exp(-u[i]))) - y[i] * u[i]
        else
            loss += log1p(exp(u[i])) - y[i] * u[i]
        end
    end

    return loss
end

""" 
    logistic_grad_hess(z, X_tilde, y)

Regresa el gradiente (g) y el Hessiano (B) de f(z)
"""

function logistic_grad_hess(z, X, y)
    n = length(y)
    u = X * z
    
    # π_i = 1 / (1 + exp(-u_i))    
    pi = 1.0 ./ (1.0 .+ exp.(-u))

    # X_tilde^T *( pi - y)
    g = X' * (pi - y)

    W = pi .* (1.0 .- pi)

    #∇^2f(z) = X^TWX
    B = (X' .* W') * X

    return g, B

end

"""
    dogleg_trust_region(f, grad_hess_f, z0; delta_max = 10.0, delta_inicial = 1.0, eta=1e-4, tol=1e-6, k_max = 1000)


Método de región de confianza implementando trayectoria DogLef.
"""

function dogleg_trust_region(f, grad_hess_f, z0; delta_max = 10.0, delta_k = 1.0, eta = 1e-4, tol = 1e-6, k_max = 1000)

    z = copy(z0)

    #Arrays del historial de valores
    hist_f = Float64[]
    hist_norm_g = Float64[]
    hist_delta = Float64[]

    start_time = time()
    k = 0


    while k < k_max
        fz = f(z)
        gk, Bk = grad_hess_f(z)
        norm_gk = norm(gk)

        #Append del historial de valores
        push!(hist_f, fz)
        push!(hist_norm_g, norm_gk)
        push!(hist_delta, delta_k)


        #Criterio de parada
        if norm_gk < tol
            return z, k, time() - start_time, hist_f, hist_norm_g, hist_delta, "Converged"
        end

        #Subproblema de Dogleg
        pk = compute_dogleg_step(gk, Bk, delta_k)

        #Reduccion real
        actual_red = fz - f(z + pk)

        #Reduccion predicha por el modelo
        predicted_red = -(dot(gk, pk) + 0.5 * dot(pk, Bk * pk))
        
        rho_k = actual_red / predicted_red

        #Actualizar delta_k
        if rho_k < 0.25
            delta_k = 0.25 * delta_k #Encogemos region
        else
            if rho_k > 0.75 && norm(pk) ≈ delta_k
                delta_k = min(2.0 * delta_k, delta_max) #Ampliamos la region al doble en caso de ser buena
            end
        end

        if rho_k > eta #Aceptar o rechazar el paso
            z+= pk
        end

        k += 1
    end

    return z, k, time() - start_time, hist_f, hist_norm_g, hist_delta, "Max iterations"
end


"""

    compute_dogleg_step(g, B, delta)

Calcula el paso pk siguiendo la trayectoria entre Cauchy y Newton
"""
function compute_dogleg_step(g, B, delta)
    lambda = 1e-6
    #Regularizacion para evitar que explote la cosa
    pb = -((B + lambda * I) \ g)
    norm_pb = norm(pb)

    if norm_pb <= delta
        return pb
    end

    alpha_c = dot(g, g) / dot(g, B * g)
    pc = -alpha_c * g
    norm_pc = norm(pc)

    if norm_pc >= delta
        return (delta / norm_pc) * pc
    end

    diff = pb - pc
    a = dot(diff, diff)
    b = 2 * dot(pc, diff)
    c = dot(pc, pc) - delta^2

    #Calculo tau
    tau = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a)

    return pc + tau * diff
end




function load_mnist_direct()
    #Cargar MNIST directamente desde MLDatasets
    train_data = MNIST(split=:train)
    test_data  = MNIST(split=:test)

    #Raw data
    X_train_raw, y_train_raw = train_data.features, train_data.targets
    X_test_raw, y_test_raw   = test_data.features, test_data.targets

    # Reshape 
    function prepare_mnist(X_raw, y_raw)
        
        #Solo tomamos 0 y 1
        mask = (y_raw .== 0) .| (y_raw .== 1)
        
        #Flatting
        X_filtered = X_raw[:, :, mask]
        X_flat = reshape(permutedims(X_filtered, (3, 1, 2)), :, 784)
        
        y_filtered = Float64.(y_raw[mask])

        # Crear matriz de diseño: añadir columna de 1s p
        n_samples = size(X_flat, 1)
        X_tilde = hcat(X_flat, ones(n_samples))

        return X_tilde, y_filtered
    end

    X_train, y_train = prepare_mnist(X_train_raw, y_train_raw)
    X_test, y_test   = prepare_mnist(X_test_raw, y_test_raw)

    return X_train, y_train, X_test, y_test
end


function classification_error(z, X_test, y_test)
    u = X_test * z
    probs = 1.0 ./ (1.0 .+ exp.(-u))
    # Predicción es 1 si prob > 0.5, 0 en otro caso 
    predictions = probs .> 0.5
    return mean(abs.(predictions .- y_test))
end



function conjugate_gradient(A, b, max_iter = 1000, tol = 1e-9)
    
    hist_norm_rk = Float64[];
    
    N = size(b);        # Dimension del problema

    k = 0
    xₖ = zeros(N);      # Initial guess
    rₖ = A*xₖ - b;      # Residuo inicial;
    pₖ = -rₖ;           # pₖ inicial
    Apₖ = similar(pₖ);  
    norm_rk = norm(rₖ);
    push!(hist_norm_rk, norm_rk);

    while (norm_rk > tol && k < max_iter)
        
        mul!(Apₖ, A, pₖ)
        numerador = dot(rₖ, rₖ);
        denominador = dot(pₖ, Apₖ)
        
        αₖ =  numerador / denominador; # αₖ = rₖᵀrₖ / pₖᵀApₖ  
        xₖ .+= αₖ .* pₖ;  
        rₖ .+= αₖ .* Apₖ;
        Βₖ = dot(rₖ, rₖ) / numerador; # Βₖ = rₖ₊₁ᵀrₖ₊₁ / rₖᵀrₖ
        pₖ = -rₖ .+ Βₖ .* pₖ;
        k += 1;
        norm_rk = norm(rₖ); 
        push!(hist_norm_rk, norm_rk);
    end

    return xₖ, hist_norm_rk, k;

end

function conjugate_gradient_precond(A, b, max_iter = 1000, tol = 1e-9)
    
    hist_norm_rk = Float64[];
    N = size(b);        # Dimension del problema

    k = 0
    xₖ = zeros(N);      # Initial guess
    M = diag(A);       # Jacobi preconditioner
    rₖ = A*xₖ - b;      # Residuo inicial;
    yₖ = rₖ ./ M      # Solving for yₖ
    pₖ = -yₖ;           # pₖ inicial
    Apₖ = similar(pₖ);  
    norm_rk = norm(rₖ);
    push!(hist_norm_rk, norm_rk);

    while (norm_rk > tol && k < max_iter)

        mul!(Apₖ, A, pₖ)
        numerador = dot(rₖ, yₖ);
        denominador = dot(pₖ, Apₖ)
        αₖ =  numerador / denominador;
        xₖ .+= αₖ .* pₖ;  
        rₖ .+= αₖ .* Apₖ;
        yₖ .= rₖ ./ M
        Βₖ = dot(rₖ, yₖ) / numerador; # Βₖ = rₖ₊₁ᵀrₖ₊₁ / rₖᵀrₖ
        pₖ .= -yₖ .+ Βₖ .* pₖ;
        k += 1;
        norm_rk = norm(rₖ); 
        push!(hist_norm_rk, norm_rk);
    end


    return xₖ, norm_rk, k;
end


function forwards_derivative(f, x, i, Δx = 1e-5)
    if Δx < 1e-19 return 0.0 end
        
    
    eᵢ = [j == i ? 1.0 : 0.0 for j in 1:length(x)]
    return (f(x .+ Δx .* eᵢ) - f(x)) / Δx

end 

function backward_derivative(f, x, i, Δx = 1e-5)
    if Δx < 1e-19 return 0.0 end
        
    
    eᵢ = [j == i ? 1.0 : 0.0 for j in 1:length(x)]
    return (f(x) - f(x .- Δx .* eᵢ)) / Δx

end 

function gradient_approximation(f, x, n, Δx = 1e-5)
    return backward_derivative.((f,), (x,), 1:n, Δx)
end


function divergence_approximation_fast(F, x, n, Δx = 1e-5)
    component = [v -> F(v)[i] for i in 1:n]
    
    ∂F = forwards_derivative.(component, (x,), 1:n, Δx)

    return sum(∂F)
end