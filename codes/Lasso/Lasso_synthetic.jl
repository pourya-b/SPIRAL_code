using LinearAlgebra
using CIAOAlgorithms
using ProximalOperators
using ProximalAlgorithms: IterationTools
using Base.Iterators: take
using CSV, DataFrames # for loading the data
using DelimitedFiles
using ProximalAlgorithms
using SparseArrays
using Random
include("../utilities/utilities.jl")
include("../utilities/comparisons_SVRG.jl")
include("../utilities/comparisons_SARAH.jl")
include("../utilities/comparisons_SGD.jl")
include("../utilities/comparisons_SAG_SAGA.jl")
cd(dirname(@__FILE__))
Random.seed!(100)
T = Float64

## dimension
n, N = 600, 300 # n:cols N:rows
# n, N = 400, 10000

############################## lasso synthesizing ##############################
λ = 1  
p = 100 # parameter to control the problem 
rho = 1 # some positive number controlling how large solution is
y_star = rand(N)
y_star ./= norm(y_star) #y^\star
C = rand(N, n) .* 2 .- 1
diag_n = dropdims(1 ./ sqrt.(sum(C .^ 2, dims = 1)), dims = 1)
C = C * spdiagm(0 => diag_n); # normalize columns

CTy = abs.(C' * y_star)
perm = sortperm(CTy, rev = true) # indices with decreasing order by abs

alpha = zeros(n)
for i = 1:n
    if i <= p
        alpha[perm[i]] = λ / CTy[perm[i]]
    else
        temp = CTy[perm[i]]
        if temp < 0.1 * λ
            alpha[perm[i]] = λ
        else
            alpha[perm[i]] = λ * rand() / temp
        end
    end
end
A = C * diagm(0 => alpha)   # scaling the columns of Cin
# generate the primal solution
x_star = zeros(n)
for i = 1:n
    if i <= p
        x_star[perm[i]] = rand() * rho / sqrt(p) * sign(dot(A[:, perm[i]], y_star))
    end
end
b = A * x_star + y_star
f_star = norm(y_star) / 2 + λ * norm(x_star, 1) # the solution
############################## lasso synthesizing ##############################

F = Vector{LeastSquares}(undef, 0) # array of f_i functions
L = Vector{T}(undef, 0) # array for Lipschitz constants
γ = Vector{T}(undef, 0) # array for stepsize constants

for i = 1:N
    tempA = A[i:i, :]
    f = LeastSquares(tempA, b[i:i], Float64(N))
    Lf = opnorm(tempA)^2 * N
    push!(F, f)
    push!(L, Lf)
    push!(γ, 0.999 * N / Lf)
end

L_ratio = Int(floor(maximum(L)/minimum(L)))
g = NormL1(λ)
func = Cost_FS(F, g, N, n, γ=γ, L=L, m=N, convex=true)


stuff = [
    Dict( # basic version (Finito/MISO )
        "LFinito" => false,   # if true: no-ls version, if false with lbfgs=false: Finito/MISO 
        "DeepLFinito" => (false, 1, 1),
        "single_stepsize" => false,
        "minibatch" => [(true, i |> Int) for i in [1]],
        "sweeping" => [1], # according to Labelsweep: 1:rnd, 2:clc, 3:sfld
        "label" => "Finito", # Finito/MISO
        "lbfgs" => false,
        "adaptive" => false,
    ),
    Dict( # no-ls
        "LFinito" => true,
        "DeepLFinito" => (false, 1, 1),
        "single_stepsize" => false,
        "minibatch" => [(true, i |> Int) for i in [1]],
        "sweeping" => [1],
        "label" => "SPIRAL-no-ls", # LFinito
        "lbfgs" => false,
        "adaptive" => false,
    ),
    Dict( # SPIRAL
        "LFinito" => false,
        "DeepLFinito" => (false, 1, 1),
        "single_stepsize" => false,
        "sweeping" => [1],
        "minibatch" => [(true, i |> Int) for i in [1]],
        "label" => "SPIRAL", # LFinito
        "lbfgs" => true,
        "adaptive" => false,
    ),
    Dict( # adaSPIRAL
        "LFinito" => false,
        "DeepLFinito" => (false, 1, 1),
        "single_stepsize" => false,
        "sweeping" => [1],
        "minibatch" => [(true, i |> Int) for i in [1]],
        "label" => "adaSPIRAL", # LFinito
        "lbfgs" => true,
        "adaptive" => true,
        "DNN" => false
    ),
]

Labelsweep = ["clc", "sfld"] # randomized, cyclical, shuffled

# run comparisons and save data to path plot_data/str
str = "test1/"
β = 1/10 # ls division parameter
options = saveplot(func, β, str)
println("test 1 for L ratio $(L_ratio)")

############################## initial point ################################
x0 = ones(n)
R = eltype(x0)
SGD_factor = N # number of grad evals per iteration for SGD
SGD_epoch = 10 # the number of SGD epochs for initializing
γ = 1/(2 * maximum(L))
solver_ = CIAOAlgorithms.SGD{R}(γ=γ)
iter_ = CIAOAlgorithms.iterator(solver_, x0, F = F, g = g, N = N)
iter_ = stopwatch(iter_)
~, ~, ~, ~, x0 = loopnsave(iter_, SGD_factor, SGD_epoch, options)
##############################################################################

epoch = 500 |> Int # maximum number of epochs (not exact for lbfgs)
convex_flag = true

println("\n****************** comparisons ******************")
~,~,sol = Comparisons_spiral!(stuff, x0, options)
Comparisons_sarah!(x0, options)
Comparisons_svrg!(x0, options)
Comparisons_sgd!(x0, options)
Comparisons_saga!(x0, options)

# quality check
# sol0 = zero(sol)
# println("\nregression quality  of  the  solution: $(0.5*norm(A*sol-b)^2), 𝓵_0: $(count(iszero, sol)), 𝓵_1: $(norm(sol,1))")
# println("regression quality of all-zero vector: $(0.5*norm(A*sol0-b)^2), 𝓵_0: $(count(iszero, sol0)), 𝓵_1: $(norm(sol0,1))")