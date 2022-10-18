using LinearAlgebra
using CIAOAlgorithms
using ProximalOperators
using ProximalAlgorithms: IterationTools
using Base.Iterators: take
using CSV, DataFrames # for loading the data
using DelimitedFiles
using ProximalAlgorithms
using Random
include("../utilities/utilities.jl")
include("../utilities/comparisons_SVRG.jl")
include("../utilities/comparisons_SARAH.jl")
include("../utilities/comparisons_SGD.jl")
include("../utilities/comparisons_SAG_SAGA.jl")
cd(dirname(@__FILE__))
Random.seed!(100)
T = Float64

data = CSV.read("../../../datasets/covtype.csv", DataFrame) # use aloi.csv, mnist_train.csv, covtype.csv, a9a.csv, UCI_sonar.txt
# data = readdlm("../../../datasets/madelon_train.data", ' ') # use for madelon_train.data
data = Matrix(data) ## if mnist_train is not scaled (by /255) or covtype is used instead of covtype.scale, SARAH is very slow!

# data = 1.0 * data[2:end,2:end] # removing labels (mnist_train)
data = 1.0 * data[2:end,1:end-1] # removing labels (covtype)
# data = 1.0 * data[1:end,4:end] # removing labels (a9a)
# data = 1.0 * hcat(data[1:end,1], data[1:end,4:end]) # removing labels (aloi)
# data = 1.0 * data[1:end,1:end-1] # removing labels (UCI_sonar.txt and madelon_train.data)

n = size(data, 2)
for i = 1:n # data normalization
    norm(data[:,i]) != 0 ? data[:,i] /= norm(data[:,i]) : data[:,i] /= 1
end

# for i = 1:size(data, 1) # wrong normalization 
#     data[i,:] /= norm(data[i,:])
# end


F = Vector{LeastSquares}(undef, 0)
L = Vector{T}(undef, 0)
γ = Vector{T}(undef, 0)

for i = 1:size(data, 1)
    if norm(data[i, :]) != 0 # to make sure there is no zero data points in the dataset
        if norm(data[i, :]) == 0
            println("wow $(i)")
        end
        f = LeastSquares(data[i:i,:], zeros(1), -1.0)
        Lf = data[i, :]' * data[i, :]
        push!(F, f)
        push!(L, Lf)
        push!(γ, 0.999 / Lf)
    else
        println("there is a zero data point! @ $(i)")
    end
end
N = length(L)
γ *= N
L_ratio = Int(floor(maximum(L)/minimum(L)))
g = IndNonnegativeBallL2()
func = Cost_FS(F, g, N, n, γ=γ, L=L, m=N, convex=false)


stuff = [
    # Dict( # basic version (Finito/MISO )
    #     "LFinito" => false,   # if true: no-ls version, if false with lbfgs=false: Finito/MISO 
    #     "DeepLFinito" => (false, 1, 1),
    #     "single_stepsize" => false,
    #     "minibatch" => [(true, i |> Int) for i in [1]],
    #     "sweeping" => [1], # according to Labelsweep: 1:rnd, 2:clc, 3:sfld
    #     "label" => "Finito", # Finito/MISO
    #     "lbfgs" => false,
    #     "adaptive" => false,
    # ),
    # Dict( # no-ls
    #     "LFinito" => true,
    #     "DeepLFinito" => (false, 1, 1),
    #     "single_stepsize" => false,
    #     "minibatch" => [(true, i |> Int) for i in [1]],
    #     "sweeping" => [1],
    #     "label" => "SPIRAL-no-ls", # LFinito
    #     "lbfgs" => false,
    #     "adaptive" => false,
    # ),
    # Dict( # SPIRAL
    #     "LFinito" => false,
    #     "DeepLFinito" => (false, 1, 1),
    #     "single_stepsize" => false,
    #     "sweeping" => [1],
    #     "minibatch" => [(true, i |> Int) for i in [1]],
    #     "label" => "SPIRAL", # LFinito
    #     "lbfgs" => true,
    #     "adaptive" => false,
    # ),
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
γ_ = 0.1
solver_ = CIAOAlgorithms.SGD{R}(γ=γ_)
iter_ = CIAOAlgorithms.iterator(solver_, x0, F = F, g = g, N = N)
iter_ = stopwatch(iter_)
~, ~, ~, ~, x0 = loopnsave(iter_, SGD_factor, SGD_epoch, options)
# ##############################################################################

epoch = 50 |> Int # maximum number of epochs (not exact for lbfgs)

println("\n****************** comparisons ******************")
~,~,sol = Comparisons_spiral!(stuff, x0, options)
# Comparisons_sarah!(x0, options)
# Comparisons_svrg!(x0, options)
# Comparisons_sgd!(x0, options)
# Comparisons_saga!(x0, options)