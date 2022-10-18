# Latafat, Themelis, Ahookhosh, Patrinos, "...."

using LinearAlgebra
using ProximalOperators
using ProximalAlgorithms.IterationTools
using ProximalAlgorithms: LBFGS, update!, mul!, reset!
using Printf
using Base.Iterators
using Random
using StatsBase: sample
using CIAOAlgorithms

import CIAOAlgorithms: solution
import ProximalOperators: prox!, prox, gradient, gradient!
export solution, solution_epoch

abstract type CIAO_iterable end

include("Bregman_Finito_basic.jl")              # Bregman Finito/MISO
include("Bregman_LFinito.jl")                   # Bregman SPIRAL-no-ls
include("Bregman_LFinito_lbfgs.jl")             # Bregman SPIRAL
include("Bregman_LFinito_lbfgs_adaptive.jl")    # Bregman adaSPIRAL


struct Bregman_Finito{R<:Real}
    γ::Maybe{Union{Array{R},R}}
    sweeping::Int8      # sweeping strategy in the inner loop, # 1:cyclical, 2:shuffled
    LBFinito::Bool
    lbfgs::Bool
    memory::Int
    minibatch::Tuple{Bool,Int}
    maxit::Int
    verbose::Bool
    freq::Int
    α::R                # in (0, 1) for stepsize
    β::R                # ls parameter for τ (interpolation between the quasi-Newton direction and the nominal step)
    η::R                # ls parameter for γ
    adaptive::Bool
    ls_tol::R           # tolerance in ls
    D::Maybe{R}         # a large number to normalize lbfgs direction if provided
    function Bregman_Finito{R}(;
        γ::Maybe{Union{Array{R},R}} = nothing,
        sweeping = 1,
        LBFinito::Bool = false,
        lbfgs::Bool = false,
        memory::Int = 5,
        minibatch::Tuple{Bool,Int} = (false, 1),
        maxit::Int = 10000,
        verbose::Bool = false,
        freq::Int = 10000,
        α::R = R(0.999),
        β::R = 1/50,
        η::R = 0.7,
        adaptive::Bool = false,
        ls_tol::R = R(1e-6),    # pretty large to avoid numerical instability!
        D::Maybe{R} = nothing,
    ) where {R}
        @assert γ === nothing || minimum(γ) > 0
        @assert maxit > 0
        @assert freq > 0
        @assert memory >= 0
        new(γ, sweeping, LBFinito, lbfgs, memory, minibatch, maxit, verbose, freq, α, β, η, adaptive, ls_tol, D)
    end
end

function (solver::Bregman_Finito{R})(
    x0::AbstractArray{C};
    F = nothing,                    # array of f_i functions
    g = ProximalOperators.Zero(),   # nonsmooth term
    H = nothing,                    # array of distance generating functions (dgf)
    L = nothing,                    # array for relative smoothness constants
    N = N,
) where {R,C<:RealOrComplex{R}}

    stop(state) = false
    disp(it, state) = @printf "%5d  \n" it

    F === nothing && (F = fill(ProximalOperators.Zero(), (N,)))
    # H === nothing && (H = fill(Poly20(), (N,)))
    # dispatching the iterator
    if solver.LBFinito # SPIRAL-no-ls
        iter = LBreg_Finito_iterable(
            F,
            g,
            H,
            x0,
            N,
            L,
            solver.γ,
            solver.sweeping,
            solver.minibatch[2],
            solver.α,
        )
    elseif solver.lbfgs 
        if solver.adaptive # adaSPIRAL
            iter = LBreg_Finito_lbfgs_adaptive_iterable(
                F,
                g,
                H,
                x0,
                N,
                L,
                solver.γ,
                solver.η,
                solver.β,
                solver.sweeping,
                solver.minibatch[2],
                solver.α,
                LBFGS(x0, solver.memory),
                solver.ls_tol,
                solver.D,
            )
        else # SPIRAL
            iter = LBreg_Finito_lbfgs_iterable(
                F,
                g,
                H,
                x0,
                N,
                L,
                solver.γ,
                solver.β,
                solver.sweeping,
                solver.minibatch[2],
                solver.α,
                LBFGS(x0, solver.memory),
                solver.ls_tol,
                solver.D,
            )
        end
    else # Finito/MISO
        iter = Breg_FINITO_basic_iterable(
            F,
            g,
            H,
            x0,
            N,
            L,
            solver.γ,
            solver.sweeping,
            solver.minibatch[2],
            solver.α,
        )
    end

    iter = take(halt(iter, stop), solver.maxit)
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
    Bregman_Finito([γ, sweeping, LBFinito, minibatch, maxit, verbose, freq, α])

Instantiate the Bregman_Finito algorithm for solving fully nonconvex optimization problems of the form
    
    minimize 1/N sum_{i=1}^N f_i(x) + g(x)

where `f_i` are smooth and `g` is possibly nonsmooth, all of which may be nonconvex.  

If `solver = Bregman_Finito(args...)`, then the above problem is solved with

	solver(x0, [F, g, H, N, L])

where F is an array containing f_i's, x0 is the initial point, and L is an array of 
smoothness moduli of f_i's; it is optional in the adaptive mode or if γ is provided. 

Optional keyword arguments are:
* `γ`: an array of N stepsizes for each coordinate 
* `sweeping::Int` 1 for uniform randomized (default), 2 for cyclic, 3 for shuffled 
* `LBFinito::Bool` low memory variant of the Bregman Finito/MISO algorithm
* `minibatch::(Bool,Int)` to use batchs of a given size    
* `maxit::Integer` (default: `10000`), maximum number of iterations to perform.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `10000`), frequency of verbosity.
* `α::R` parameter where γ_i = αN/L_i
"""

Bregman_Finito(::Type{R}; kwargs...) where {R} = Bregman_Finito{R}(; kwargs...)
Bregman_Finito(; kwargs...) = Bregman_Finito(Float64; kwargs...)


"""
If `solver = Bregman_Finito(args...)`, then 

    itr = iterator(solver, [F, g, H, x0, N, L])

is an iterable object. Note that [maxit, verbose, freq] fields of the solver are ignored here. 

The solution at any given state can be obtained using solution(state), e.g., 
for state in Iterators.take(itr, maxit)
    # do something using solution(state)
end

See https://docs.julialang.org/en/v1/manual/interfaces/index.html 
and https://docs.julialang.org/en/v1/base/iterators/ for a list of iteration utilities
"""


function iterator(
    solver::Bregman_Finito{R},
    x0::AbstractArray{C};
    F = nothing,                    # array of f_i functions
    g = ProximalOperators.Zero(),   # nonsmooth term
    H = nothing,                    # array of distance generating functions (dgf)
    L = nothing,                    # array for relative smoothness constants
    N = N,
) where {R,C<:RealOrComplex{R}}
    F === nothing && (F = fill(ProximalOperators.Zero(), (N,)))
    H === nothing && (H = fill(Poly20(), (N,)))
    # dispatching the iterator
    if solver.LBFinito # SPIRAL-no-ls
        iter = LBreg_Finito_iterable(
            F,
            g,
            H,
            x0,
            N,
            L,
            solver.γ,
            solver.sweeping,
            solver.minibatch[2],
            solver.α,
        )
    elseif solver.lbfgs 
        if solver.adaptive # adaSPIRAL
            iter = LBreg_Finito_lbfgs_adaptive_iterable(
                F,
                g,
                H,
                x0,
                N,
                L,
                solver.γ,
                solver.η,
                solver.β,
                solver.sweeping,
                solver.minibatch[2],
                solver.α,
                LBFGS(x0, solver.memory),
                solver.ls_tol,
                solver.D,
            )
        else # SPIRAL
            iter = LBreg_Finito_lbfgs_iterable(
                F,
                g,
                H,
                x0,
                N,
                L,
                solver.γ,
                solver.β,
                solver.sweeping,
                solver.minibatch[2],
                solver.α,
                LBFGS(x0, solver.memory),
                solver.ls_tol,
                solver.D,
            )
        end
    else # Finito/MISO
        iter = Breg_FINITO_basic_iterable(
            F,
            g,
            H,
            x0,
            N,
            L,
            solver.γ,
            solver.sweeping,
            solver.minibatch[2],
            solver.α,
        )
    end

    return iter
end

######TODO:
# default value for H other the adaSPIRAL
# revisit adaptive version
# prox_Breg currently accepts one H only! 