using LinearAlgebra
using Statistics
using Pickle
using GZip
using PyCall
using MLDatasets
using DelimitedFiles
using Images
using FileIO
using ImageView
using Plots

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

    hist_f = Float64[]
    hist_alpha = Float64[]
    hist_norm_g = Float64[]
    hist_backtracks = Int[]

    start_time = time();
    k = 0;
    stop_r = "max_iter";
    
    while k < k_max
        fx = f(x)
        gx = grad_f(x)
        ng = norm(gx, 2)

        push!(hist_f, fx)
        push!(hist_norm_g, ng)

        if ng <= tol
            stop_r = "||∇f|| ≈ 0"
            break
        end

        pk = -gx
        alpha = 1.0
        n_backtracks = 0

        while f(x + alpha * pk) > fx + c1 * alpha * dot(gx, pk)
            alpha *= rho
            n_backtracks += 1
            if alpha < 1e-16
                stop_r = "alpha_min"
                break
            end
        end

        if (stop_r) == "alpha_min"; break; end

        x .+= alpha * pk
        push!(hist_alpha, alpha)
        push!(hist_backtracks, n_backtracks)
        
        k += 1
    end

    total_time = time() - start_time
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
rosenbrock(x)

Generador del valor de la función de Rosenbrock en base a un vector inicial.

# Parámetros:
- `x`: Vector de dimensión N.

# Retorno:
- `suma`: Evaluación de la función.
"""
function rosenbrock(x)
    n = length(x)
    suma = 0.0
    for i in 1:(n-1)
        suma += 100 * (x[i + 1] - x[i]^2)^2 + (1 - x[i])^2
    end
    return suma
end 

""" 
grad_rosenbrock(x)

Calcula el gradiente de la función de Rosenbrock.

# Parámetros:
- `x`: Vector de dimensión N.

# Retorno:
- `g`: Vector gradiente evaluado en `x`.
"""
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
flatten(Z)

Aplana una matriz bidimensional en un vector unidimensional.

# Parámetros:
- `Z`: Matriz a aplanar.

# Retorno:
- Vector aplanado.
"""
flatten(Z) = vec(Z)

"""
unflatten(z_vec)

Restaura la dimensionalidad de un vector aplanado a una matriz N_samples x P_dim.
Asume que N_samples y P_dim existen en el entorno global.

# Parámetros:
- `z_vec`: Vector aplanado.

# Retorno:
- Matriz redimensionada.
"""
unflatten(z_vec) = reshape(z_vec, N_samples, P_dim)

"""
stress_mds(z_vec, N_samples, D_target)

Calcula la pérdida para MDS (Stress).
Representa la discrepancia entre las distancias en el espacio original y el proyectado.

# Parámetros:
- `z_vec`: Vector aplanado de coordenadas en baja dimensión (N*P).
- `N_samples`: Número total de puntos.
- `D_target`: Matriz de distancias euclidianas objetivo.

# Retorno:
- `loss`: Valor escalar de la pérdida de Stress.
"""
function stress_mds(z_vec, N_samples, D_target) 
    Z = unflatten(z_vec)
    loss = 0.0
    for i in 1:N_samples
        for j in (i + 1):N_samples
            dist_z = norm(Z[i, :] - @view Z[j, :])
            loss += (D_target[i, j] - dist_z)^2
        end
    end
    return 0.5 * loss
end 

"""
grad_stress_mds(z_vec, N_samples, P_dim, D_target)

Calcula el gradiente analítico de la función Stress respecto a las posiciones Z.

# Parámetros:
- `z_vec`: Vector aplanado de coordenadas.
- `N_samples`: Número total de puntos.
- `P_dim`: Dimensión del espacio de proyección (típicamente 2).
- `D_target`: Matriz de distancias euclidianas objetivo.

# Retorno:
- Vector aplanado del gradiente.
"""
function grad_stress_mds(z_vec, N_samples, P_dim, D_target)
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

# Parámetros:
- `X`: Matriz de datos donde cada fila es una observación.

# Retorno:
- `D`: Matriz simétrica de distancias.
"""
function distances_matrix(X)
    n = size(X, 1)
    D = zeros(n, n)
    for i in 1:n, j in 1:n
        D[i, j] = norm(X[i, :] - X[j, :])
    end
    return D
end

"""
logistic_loss(z, X, y)

Calcula la log-likelihood negativa f(z) para la regresión logística.

# Parámetros:
- `z`: Vector de pesos.
- `X`: Matriz de diseño.
- `y`: Vector de etiquetas {0, 1}.

# Retorno:
- `loss`: Valor de la pérdida evaluada.
"""
function logistic_loss(z, X, y)
    n = length(y)
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
logistic_grad_hess(z, X, y)

Calcula el gradiente y el Hessiano de f(z) para regresión logística.

# Parámetros:
- `z`: Vector de pesos.
- `X`: Matriz de diseño.
- `y`: Vector de etiquetas {0, 1}.

# Retorno:
- `g`: Vector gradiente.
- `B`: Matriz Hessiana.
"""
function logistic_grad_hess(z, X, y)
    n = length(y)
    u = X * z
    pi = 1.0 ./ (1.0 .+ exp.(-u))
    g = X' * (pi - y)
    W = pi .* (1.0 .- pi)
    B = (X' .* W') * X
    return g, B
end

"""
dogleg_trust_region(f, grad_hess_f, z0; delta_max, delta_k, eta, tol, k_max)

Método de región de confianza implementando trayectoria Dogleg.

# Parámetros:
- `f`: Función objetivo.
- `grad_hess_f`: Función que retorna el gradiente y el Hessiano.
- `z0`: Punto inicial.
- `delta_max`: Radio máximo permitido.
- `delta_k`: Radio inicial.
- `eta`: Umbral de aceptación del paso.
- `tol`: Tolerancia de paro para el gradiente.
- `k_max`: Número máximo de iteraciones.

# Retorno:
- `z`: Punto final.
- `k`: Iteraciones completadas.
- Tiempo de ejecución.
- `hist_f`, `hist_norm_g`, `hist_delta`: Historiales de la función, gradiente y radio.
- Razón de terminación.
"""
function dogleg_trust_region(f, grad_hess_f, z0; delta_max = 10.0, delta_k = 1.0, eta = 1e-4, tol = 1e-6, k_max = 1000)
    z = copy(z0)
    hist_f = Float64[]
    hist_norm_g = Float64[]
    hist_delta = Float64[]
    start_time = time()
    k = 0

    while k < k_max
        fz = f(z)
        gk, Bk = grad_hess_f(z)
        norm_gk = norm(gk)

        push!(hist_f, fz)
        push!(hist_norm_g, norm_gk)
        push!(hist_delta, delta_k)

        if norm_gk < tol
            return z, k, time() - start_time, hist_f, hist_norm_g, hist_delta, "Converged"
        end

        pk = compute_dogleg_step(gk, Bk, delta_k)
        actual_red = fz - f(z + pk)
        predicted_red = -(dot(gk, pk) + 0.5 * dot(pk, Bk * pk))
        rho_k = actual_red / predicted_red

        if rho_k < 0.25
            delta_k = 0.25 * delta_k
        else
            if rho_k > 0.75 && norm(pk) ≈ delta_k
                delta_k = min(2.0 * delta_k, delta_max)
            end
        end

        if rho_k > eta
            z += pk
        end
        k += 1
    end

    return z, k, time() - start_time, hist_f, hist_norm_g, hist_delta, "Max iterations"
end

"""
compute_dogleg_step(g, B, delta)

Calcula el paso pk siguiendo la trayectoria entre el paso de Cauchy y el paso de Newton.

# Parámetros:
- `g`: Gradiente evaluado.
- `B`: Hessiano evaluado.
- `delta`: Radio actual de la región de confianza.

# Retorno:
- Paso propuesto `pk`.
"""
function compute_dogleg_step(g, B, delta)
    lambda = 1e-6
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
    tau = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a)

    return pc + tau * diff
end

"""
load_mnist_direct()

Carga el conjunto de datos MNIST, filtra dígitos 0 y 1, y prepara la matriz de diseño.

# Parámetros:
- Ninguno.

# Retorno:
- `X_train`, `y_train`, `X_test`, `y_test`: Matrices de características aplanadas y vectores de etiquetas.
"""
function load_mnist_direct()
    train_data = MNIST(split=:train)
    test_data  = MNIST(split=:test)

    X_train_raw, y_train_raw = train_data.features, train_data.targets
    X_test_raw, y_test_raw   = test_data.features, test_data.targets

    function prepare_mnist(X_raw, y_raw)
        mask = (y_raw .== 0) .| (y_raw .== 1)
        X_filtered = X_raw[:, :, mask]
        X_flat = reshape(permutedims(X_filtered, (3, 1, 2)), :, 784)
        y_filtered = Float64.(y_raw[mask])
        n_samples = size(X_flat, 1)
        X_tilde = hcat(X_flat, ones(n_samples))

        return X_tilde, y_filtered
    end

    X_train, y_train = prepare_mnist(X_train_raw, y_train_raw)
    X_test, y_test   = prepare_mnist(X_test_raw, y_test_raw)

    return X_train, y_train, X_test, y_test
end

"""
classification_error(z, X_test, y_test)

Calcula el error de clasificación promedio para el modelo de regresión logística.

# Parámetros:
- `z`: Vector de pesos aprendidos.
- `X_test`: Matriz de diseño de prueba.
- `y_test`: Etiquetas verdaderas de prueba.

# Retorno:
- Error medio como valor flotante.
"""
function classification_error(z, X_test, y_test)
    u = X_test * z
    probs = 1.0 ./ (1.0 .+ exp.(-u))
    predictions = probs .> 0.5
    return mean(abs.(predictions .- y_test))
end

"""
conjugate_gradient(A, b, max_iter, tol)

Algoritmo iterativo de Gradiente Conjugado para resolver sistemas de ecuaciones lineales Ax = b,
donde A es una matriz definida positiva simétrica.

# Parámetros:
- `A`: Matriz del sistema.
- `b`: Vector del lado derecho.
- `max_iter`: Número máximo de iteraciones.
- `tol`: Tolerancia de convergencia.

# Retorno:
- `xₖ`: Solución aproximada.
- `hist_norm_rk`: Historial de la norma del residuo.
- `k`: Número total de iteraciones tomadas.
"""
function conjugate_gradient(A, b, max_iter = 1000, tol = 1e-9)
    hist_norm_rk = Float64[];
    N = size(b);

    k = 0
    xₖ = zeros(N);
    rₖ = A*xₖ - b;
    pₖ = -rₖ;
    Apₖ = similar(pₖ);
    norm_rk = norm(rₖ);
    push!(hist_norm_rk, norm_rk);

    while (norm_rk > tol && k < max_iter)
        mul!(Apₖ, A, pₖ)
        numerador = dot(rₖ, rₖ);
        denominador = dot(pₖ, Apₖ)
        
        αₖ =  numerador / denominador;
        xₖ .+= αₖ .* pₖ;  
        rₖ .+= αₖ .* Apₖ;
        Βₖ = dot(rₖ, rₖ) / numerador;
        pₖ = -rₖ .+ Βₖ .* pₖ;
        k += 1;
        norm_rk = norm(rₖ); 
        push!(hist_norm_rk, norm_rk);
    end

    return xₖ, hist_norm_rk, k;
end

"""
conjugate_gradient_precond(A, b, max_iter, tol)

Algoritmo de Gradiente Conjugado Precondicionado utilizando el precondicionador de Jacobi
(diagonal de A) para mejorar la tasa de convergencia.

# Parámetros:
- `A`: Matriz del sistema.
- `b`: Vector del lado derecho.
- `max_iter`: Número máximo de iteraciones.
- `tol`: Tolerancia de convergencia.

# Retorno:
- `xₖ`: Solución aproximada.
- `hist_norm_rk`: Historial de la norma del residuo.
- `k`: Número de iteraciones completadas.
"""
function conjugate_gradient_precond(A, b, max_iter = 1000, tol = 1e-9)
    hist_norm_rk = Float64[];
    N = size(b);

    k = 0
    xₖ = zeros(N);
    M = diag(A);
    rₖ = A*xₖ - b;
    yₖ = rₖ ./ M
    pₖ = -yₖ;
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
        Βₖ = dot(rₖ, yₖ) / numerador;
        pₖ .= -yₖ .+ Βₖ .* pₖ;
        k += 1;
        norm_rk = norm(rₖ); 
        push!(hist_norm_rk, norm_rk);
    end

    return xₖ, hist_norm_rk, k;
end

"""
forwards_derivative_img(U, x, y, dim)

Calcula la derivada hacia adelante de una imagen en un pixel y dimensión específicos.

# Parámetros:
- `U`: Matriz de la imagen.
- `x`, `y`: Coordenadas del pixel.
- `dim`: Dimensión (1 para x, 2 para y).

# Retorno:
- Valor de la derivada numérica o 0.0 en caso de frontera.
"""
function forwards_derivative_img(U, x, y, dim)
    M, N = size(U)
    if dim == 1
        return x < M ? U[x+1, y] - U[x,y] : 0.0
    else
        return y < N ? U[x, y+1] - U[x,y] : 0.0
    end
end 

"""
backward_derivative_img(V, x, y, dim)

Calcula la derivada hacia atrás de un campo/matriz en un pixel y dimensión específicos.

# Parámetros:
- `V`: Matriz de datos.
- `x`, `y`: Coordenadas del pixel.
- `dim`: Dimensión (1 para x, 2 para y).

# Retorno:
- Valor de la derivada numérica o 0.0 en caso de frontera.
"""
function backward_derivative_img(V, x, y, dim)
    if dim == 1
        return x > 1 ? V[x, y] - V[x-1, y] : 0.0
    else       
        return y > 1 ? V[x, y] - V[x, y-1] : 0.0
    end
end

"""
gradient_approximation_img(U, x, y)

Aproxima el gradiente bidimensional de una imagen en un pixel usando diferencias hacia adelante.

# Parámetros:
- `U`: Matriz de la imagen.
- `x`, `y`: Coordenadas del pixel.

# Retorno:
- Arreglo con las componentes `[dx, dy]`.
"""
function gradient_approximation_img(U, x, y)
    dx = forwards_derivative_img(U, x, y, 1)
    dy = forwards_derivative_img(U, x, y, 2)
    return [dx, dy]
end

"""
divergence_approximation_img(V_x, V_y, x, y)

Aproxima la divergencia de un campo vectorial bidimensional usando diferencias hacia atrás.

# Parámetros:
- `V_x`: Matriz componente X del campo vectorial.
- `V_y`: Matriz componente Y del campo vectorial.
- `x`, `y`: Coordenadas de evaluación.

# Retorno:
- Valor escalar de la divergencia en el pixel dado.
"""
function divergence_approximation_img(V_x, V_y, x, y)
    dx = backward_derivative_img(V_x, x, y, 1)
    dy = backward_derivative_img(V_y, x, y, 2)
    return dx + dy
end

"""
denoising(u0_matrix, total_iters)

Aplica un algoritmo de regularización de variación total (TV) para la reducción de ruido en imágenes.

# Parámetros:
- `u0_matrix`: Matriz de la imagen ruidosa original.
- `total_iters`: Cantidad total de iteraciones a ejecutar.

# Retorno:
- `u`: Imagen reconstruida (filtrada).
- `hist_norm_∇u`: Historial de la norma del gradiente
"""
function denoising(u0_matrix, total_iters)
    u0 = Float64.(u0_matrix)
    if maximum(u0) <= 1.0
        u0 .= u0 .* 255.0
    end
    
    M, N = size(u0)
    u = copy(u0)

    α = 0.0001
    γ = 40.0
    η = 0.0001
    iters = total_iters

    hist_norm_∇u = Float64[]
    dx = zeros(M, N)
    dy = zeros(M, N)
    norm_∇u = zeros(M, N)
    div_V = zeros(M, N)
    push!(hist_norm_∇u, norm(norm_∇u))

    for k in 1:iters
        dx .= 0.0 
        dy .= 0.0
        
        @views dx[1:end-1, :] .= u[2:end, :] .- u[1:end-1, :]
        @views dy[:, 1:end-1] .= u[:, 2:end] .- u[:, 1:end-1]

        @. norm_∇u = sqrt(dx^2 + dy^2 + η)
        push!(hist_norm_∇u, norm(norm_∇u))
       
        @. dx = dx / norm_∇u  
        @. dy = dy / norm_∇u 

        div_V .= 0.0
        
        @views div_V[2:end, :] .+= dx[2:end, :] .- dx[1:end-1, :]
        @views div_V[:, 2:end] .+= dy[:, 2:end] .- dy[:, 1:end-1]

        @. u = u - α * ((u - u0) - γ * div_V)

        if k % 1000 == 0
            println("Iteración $k completada")
            u_visual = clamp.(u ./ 255.0, 0.0, 1.0) 
            display(colorview(Gray, u_visual))
        end
    end

    return u, hist_norm_∇u
end