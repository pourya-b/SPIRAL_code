using LinearAlgebra
using BregmanBC
using CIAOAlgorithms
using ProximalOperators
using ProximalAlgorithms: IterationTools
using Base.Iterators: take
using CSV, DataFrames # for loading the data
using DelimitedFiles
using ProximalAlgorithms
using SparseArrays
using Images, TestImages
using Hadamard, Statistics
using Random

include("../utilities/utilities.jl")
include("../utilities/utilities_breg.jl")
include("../utilities/comparisons_SMD.jl")

cd(dirname(@__FILE__))
R = Float64
rndseed = 50
Random.seed!(rndseed)

digits = "digits_6"
img = readdlm(string("../../../datasets/", digits, ".csv"), ',', Float64, '\n')

image_num = 100 # the image in the dataset
p_fail = 0.02 # the probability of corruption 

#####----------------------image examples--------------------------------------
x_star = -img[image_num, :] #the image is vectorized, -1 is to make the background white
n = length(x_star)
k = 5 # from Duchi's Sec 6.3
N = k * n 
# generate A and b
A, b = Ab_image(x_star,k, p_fail) # random
# initial point according to Wang!
x0 =  initializeX(A, b, N) # initialization

#### ------------------------------------ init point visualization ---------------------------------
mkpath(string("solutions/",digits))
Gray.(reshape(x0, 16,16))
x_clamp = map(clamp01nan, x0)
save(string("solutions/",digits,"/initialization.png"), colorview(Gray, reshape(abs.(x_clamp), 16, 16)'))
Gray.(reshape(x_star, 16,16))
x_clamp = map(clamp01nan, x_star)
save(string("solutions/",digits,"/original.png"), colorview(Gray, reshape(abs.(x_clamp), 16, 16)'))
#### ---------------------------------------------------------------------------------------------

F = Vector{Quartic}(undef, 0) # array of f_i functions
H = Vector{Poly42}(undef, 0) # array of distance generating functions (dgf)
L = Vector{R}(undef, 0) # array for relative smoothness constants
γ = Vector{R}(undef, 0) # array for stepsize constants

for i = 1:N
    tempA = A[i, :]
    f = Quartic(tempA, b[i])
    normA = norm(tempA)
    Lf = 3 * normA^4 + normA^2 * abs(b[i])
    h = Poly42()
    push!(F, f)
    push!(H, h)
    push!(L, Lf)
    push!(γ, 0.999 * R(N) / Lf)
end

L_ratio = Int(floor(maximum(L)/minimum(L)))
λ = 0.5/N # for NormL1(λ)
g = NormL1(λ)
func = Cost_FS(F, g, N, n, H=H, γ=γ, L=L, m=N, convex=false)

stuff_euclidean = [
    Dict( # adaSPIRAL (Euclidean)
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

stuff_breg = [
    Dict( # basic version (Bregman Finito/MISO)
        "LFinito" => false,   # if true: no-ls version, if false with lbfgs=false: Finito/MISO 
        "single_stepsize" => false,
        "minibatch" => [(true, i |> Int) for i in [1]],
        "sweeping" => [1], # according to Labelsweep: 1:clc, 2:sfld
        "label" => "Finito", # Finito/MISO
        "lbfgs" => false,
        "adaptive" => false,
    ),
    Dict( # Bregman SPIRAL-no-ls
        "LFinito" => true,   # if true: no-ls version, if false with lbfgs=false: Finito/MISO 
        "single_stepsize" => false,
        "minibatch" => [(true, i |> Int) for i in [1]],
        "sweeping" => [1], # according to Labelsweep: 1:clc, 2:sfld
        "label" => "BSPIRAL-no-ls", 
        "lbfgs" => false,
        "adaptive" => false,
    ),
    Dict( # Bregman SPIRAL
        "LFinito" => false,   # if true: no-ls version, if false with lbfgs=false: Finito/MISO 
        "single_stepsize" => false,
        "minibatch" => [(true, i |> Int) for i in [1]],
        "sweeping" => [1], # according to Labelsweep: 1:clc, 2:sfld
        "label" => "BSPIRAL", 
        "lbfgs" => true,
        "adaptive" => false,
    ),
    Dict( # Bregman adaSPIRAL
        "LFinito" => false,   # if true: no-ls version, if false with lbfgs=false: Finito/MISO 
        "single_stepsize" => false,
        "minibatch" => [(true, i |> Int) for i in [1]],
        "sweeping" => [1], # according to Labelsweep: 1:clc, 2:sfld
        "label" => "adaBSPIRAL", 
        "lbfgs" => true,
        "adaptive" => true,
    ),
]

Labelsweep = ["clc", "sfld"] # randomized, cyclical, shuffled

# run comparisons and save data to path plot_data/str
str = digits
β = 1/10 # ls division parameter
println("test 1 for L ratio $(L_ratio) ")
options = saveplot(func, β, str, epoch_per_iter=4)

epoch = 550 |> Int # maximum number of epochs (not exact for lbfgs)

println("\n****************** comparisons ******************")
# Comparisons_spiral_eucl!(stuff_euclidean, x0, options)
# Comparisons_spiral_breg!(stuff_breg, x0, options)
Comparisons_smd!(x0, options)