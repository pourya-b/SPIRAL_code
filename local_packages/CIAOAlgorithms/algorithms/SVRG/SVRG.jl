# Xiao, Zhang, "A proximal stochastic gradient method  with progressive
# variance reduction.", SIAM Journal on Optimization 24.4 (2014): 2057-2075.
# 
# Reddi, Sra, Poczos, and Smola, "Proximal stochastic methods for nonsmooth
# nonconvex finite-sum optimization." In Advances in Neural Information 
# Processing Systems (2016), pp. 1145–1153.
#
# Allen-Zhu, Yuan, "Improved SVRG for non-strongly-convex or 
# sum-of-non-convex objectives." In Proceedings of the 33rd 
# International Conference on Machine Learning (2016): 1080–1089.
#

using LinearAlgebra
using ProximalOperators
using ProximalAlgorithms.IterationTools
using Printf
using Base.Iterators
using Random
# using Flux # for DNN training

export solution

include("SVRG_prox.jl")
# include("SVRG_basic_DNN.jl")

struct SVRG{R<:Real}
    γ::Maybe{R}
    maxit::Int
    verbose::Bool
    freq::Int
    m::Maybe{Int}  # number of epochs for inner loop 
    DNN::Bool
    function SVRG{R}(;
        γ::Maybe{R} = nothing,
        maxit::Int = 10000,
        verbose::Bool = false,
        freq::Int = 1000,
        m::Maybe{Int} = nothing,
        DNN::Bool = false,
    ) where {R}
        @assert γ === nothing || γ > 0
        @assert maxit > 0
        @assert freq > 0
        new(γ, maxit, verbose, freq, m, DNN)
    end
end

function (solver::SVRG{R})(
    x0::AbstractArray{C};
    F = nothing,
    g = ProximalOperators.Zero(),
    L = nothing,
    μ = nothing,
    N = N,
) where {R,C<:RealOrComplex{R}}

    stop(state::SVRG_prox_state) = false
    disp(it, state) = @printf "%5d | %.3e  \n" it state.γ

    F === nothing && (F = fill(ProximalOperators.Zero(), (N,)))
    solver.m === nothing ? m = N : m = solver.m

    # dispatching the structure
    iter = SVRG_prox_iterable(F, g, x0, N, L, μ, solver.γ, m, solver.plus)
    iter = take(halt(iter, stop), maxit)
    iter = enumerate(iter)
    num_iters, state_final = nothing, nothing
    for (it_, state_) in iter  # unrolling the iterator 
        # see https://docs.julialang.org/en/v1/manual/interfaces/index.html
        if solver.verbose && mod(it_, solver.freq) == 0
            disp(it_, state_)
        end
        num_iters, state_final = it_, state_
    end
    if solver.verbose && mod(num_iters, solver.freq) !== 0
        disp(num_iters, state_final)
    end # for the final iteration
    return solution(state_final), num_iters
end


"""
    SVRG([γ, maxit, verbose, freq, m])

Instantiate the SVRG algorithm  for solving (strongly) convex optimization problems of the form
    
    minimize 1/N sum_{i=1}^N f_i(x) + g(x) 

If `solver = SVRG(args...)`, then the above problem is solved with

	solver(x0, [F, g, N, L, μ])

where F is an array containing f_i's, x0 is the initial point, and L, μ are arrays of 
smoothness and strong convexity moduli of f_i's; they are optional when γ is provided.  

Optional keyword arguments are:
* `γ`: stepsize  
* `L`: an array of smoothness moduli of f_i's 
* `μ`: (if strongly convex) an array of strong convexity moduli of f_i's 
* `maxit::Integer` (default: `10000`), maximum number of iterations to perform.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `100`), frequency of verbosity.

"""

SVRG(::Type{R}; kwargs...) where {R} = SVRG{R}(; kwargs...)
SVRG(; kwargs...) = SVRG(Float64; kwargs...)


"""
If `solver = SVRG(args...)`, then 

    itr = iterator(solver, x0, [F, g, N, L, μ])

is an iterable object. Note that [maxit, verbose, freq] fields of the solver are ignored here. 

The solution at any given state can be obtained using solution(state), e.g., 
for state in Iterators.take(itr, maxit)
    # do something using solution(state)
end

See https://docs.julialang.org/en/v1/manual/interfaces/index.html 
and https://docs.julialang.org/en/v1/base/iterators/ for a list of iteration utilities
"""

function iterator(
    solver::SVRG{R},
    x0::AbstractArray{C};
    F = nothing,
    g = ProximalOperators.Zero(),
    L = nothing,
    μ = nothing,
    N = N,
) where {R,C<:RealOrComplex{R}}
    F === nothing && (F = fill(ProximalOperators.Zero(), (N,)))
    solver.m === nothing ? m = N : m = solver.m
    # dispatching the iterator
    if solver.DNN
        iter = SVRG_basic_DNN_iterable(F, g, x0, N, L, μ, solver.γ, m)
    else
        iter = SVRG_prox_iterable(F, g, x0, N, L, μ, solver.γ, m)
    end

    return iter
end
