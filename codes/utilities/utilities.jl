# utility for SPIRAL. It compares different versions of the SPIRAL according to the stuff list in Comparisons function.
# function call map: Comparisons --> loopnsave --> eval_res 

struct Cost_FS{Tf,Tg,Mx,Cf,Th}
    F::Union{Tf, Array{Tf}}             # smooth term
    g::Tg                               # nonsmooth term
    N::Int                              # number of data points in the finite sum problem
    n::Int                              # F[i] : R^n \to R
    H::Union{Th, Array{Th}}             # distance generating function 
    γ::Union{Array{Float64}, Float64, Nothing} # stepsize
    L::Union{Float64,Array{Float64},Nothing}   # Lipschitz constant
    m::Int                              # number of iterates in the inner loop
    DNN::Bool                           # if the DNN version should be activated
    convex::Bool                        # if the convex mode should be activated
    data::Mx                            # a matrix containing data 
    configure_func::Cf                  # a function to process DNN parameters (TODO: move to CIAO utilities)
    function Cost_FS(
        F::Union{Tf, Array{Tf}}, g::Tg, N::Int, n::Int;
        H::Union{Th, Array{Th}} = nothing, γ::Union{Array{Float64}, Float64, Nothing} = nothing, L::Union{Float64,Array{Float64},Nothing} = nothing, m::Int = N, DNN::Bool = false, convex::Bool = false, data::Mx = nothing, configure_func::Cf = nothing, 
    ) where{Tf,Tg,Mx,Cf,Th}
        new{Tf,Tg,Mx,Cf,Th}(F, g, N, n, H, γ, L, m, DNN, convex, data, configure_func)
    end
end

function (S::Cost_FS)(x::Array{R}) where {R <: Real} # cost function value 
    cost = S.g(x)
    for i = 1:N
        cost += S.F[i](x) / S.N
    end
    return cost
end

function (S::Cost_FS)(x::Mx) where {Mx}
    w = DNN_config!()
    cost = S.g(w)
    cost += S.F(batchmemaybe(x)...)
    return cost
end

struct saveplot{Tf,Ry,Tx}
    func::Tf                   # cost_FS handle
    β::Ry                   # ls division parameter
    str::String             # name of the test
    x_star::Tx
    epoch_per_iter::Int     # epoch per iteration 
    function saveplot(
        func::Tf, β::Ry, str::String;
        x_star::Tx = nothing, 
        epoch_per_iter::Int = 3, # with efficient implementation for problems such as lasso, NNPCA, logistic regression
    ) where{Tf,Ry,Tx}
        new{Tf,Ry,Tx}(func, β, str, x_star, epoch_per_iter)
    end
end

function Comparisons_spiral!(
    stuff,
    x0,
    options,
) 
    func = options.func
    n = func.n
    N = func.N
    L = func.L

    # for saving and returning
    cost_history = Vector{Vector{T}}(undef, 0)
    res_history = Vector{Vector{T}}(undef, 0)
    it_history = Vector{Vector{T}}(undef, 0)
    cpu_history = Vector{Vector{T}}(undef, 0)

    cnt = 0 # counter
    t1 = length(stuff)
    sol = copy(x0)
    for i = 1:t1 # loop for all the matrials in stuff
        t2 = length(stuff[i]["sweeping"])
        
        for j = 1:t2 # loop over index update strategy between ["clc", "sfld"]
            LFinito = stuff[i]["LFinito"]
            DeepLFinito = stuff[i]["DeepLFinito"]
            lbfgs = stuff[i]["lbfgs"]
            adaptive = stuff[i]["adaptive"]
            single_stepsize = stuff[i]["single_stepsize"]
            label = stuff[i]["label"]
            sweeping = stuff[i]["sweeping"][j]
            t3 = length(stuff[i]["minibatch"])

            for l = 1:t3 # loop for minibatch numbers
                cnt += 1
                minibatch = stuff[i]["minibatch"][l]
                size_batch = minibatch[2]
                println("\n\nsolving using $(label) with $(Labelsweep[sweeping]) sweeping and batchsize $(size_batch)"); flush(stdout)

                solver = CIAOAlgorithms.Finito{T}( # returns the suitable solver (SPIRAL version) based on the parameters
                    sweeping = sweeping,
                    LFinito = LFinito,
                    DeepLFinito = DeepLFinito,
                    minibatch = minibatch,
                    lbfgs = lbfgs,
                    adaptive = adaptive,
                    β = options.β,
                    ls_tol = 1e-6,  # the tolerance for the backtrackings
                )

                L_s = nothing # L vector, adjusted for single or multi stepsizes
                if L != nothing
                    if single_stepsize
                        L_s = maximum(L)   # has to be divided since 1/L_F = \bar γ = (\sum 1/γ_i)^-1 = γ/N
                    else
                        L_s = L
                    end
                end

                factor = N # number of grad evals per iteration (equal to N for default 'Finito/MISO')
                if LFinito # low memory with no ls
                    if DeepLFinito[1]
                        factor = N * (1 + DeepLFinito[3])
                    else
                        factor = (options.epoch_per_iter - 1) * N
                    end
                elseif lbfgs
                    if DeepLFinito[1]
                        factor = (2 +  DeepLFinito[3]) * N
                    else
                        factor = (options.epoch_per_iter) * N
                    end
                end

                Maxit = Int(ceil(epoch * N/ factor)) # outer iteration number needed for the algorithm
                iter = CIAOAlgorithms.iterator( # iterable handle to pass to the for loop
                    solver,
                    x0,
                    F = func.F,
                    g = func.g,
                    L = L_s,
                    N = N,
                )

                iter = stopwatch(iter)
                it_hist, cost_hist, res_hist, cpu_hist, sol =
                        loopnsave(iter, factor, Maxit, options) # the main iterations using the iterable 'iter'

                # for returning
                push!(cost_history, cost_hist)
                push!(res_history, res_hist)
                push!(it_history, it_hist)
                push!(cpu_history, cpu_hist)

                # for saving
                output = [it_hist cost_hist]
                mkpath(string("plot_data/",str,"/cost"))
                open(
                    string(
                        "plot_data/",
                        str,
                        "cost/",
                        "N_",
                        N,
                        "_n_",
                        n,
                        "_batch_",
                        size_batch, # for l
                        "_",
                        label, # for i
                        "_",
                        sweeping, # for j
                        "_Lratio_",
                        L_ratio,
                        ".txt",
                    ),
                    "w",
                ) do io
                    writedlm(io, output)
                end

                # residual |z- prox(z)|
                output = [it_hist res_hist]               
                mkpath(string("plot_data/",str,"/res"))
                open(
                    string(
                        "plot_data/",
                        str,
                        "res/",
                        "N_",
                        N,
                        "_n_",
                        n,
                        "_batch_",
                        size_batch,
                        "_",
                        label,
                        "_",
                        sweeping,
                        "_Lratio_",
                        L_ratio,
                        ".txt",
                    ),
                    "w",
                ) do io
                    writedlm(io, output)
                end

                output = [cpu_hist res_hist]               
                mkpath(string("plot_data/",str,"/cpu"))
                open(
                    string(
                        "plot_data/",
                        str,
                        "cpu/",
                        "N_",
                        N,
                        "_n_",
                        n,
                        "_batch_",
                        size_batch,
                        "_",
                        label,
                        "_",
                        sweeping,
                        "_Lratio_",
                        L_ratio,
                        ".txt",
                    ),
                    "w",
                ) do io
                    writedlm(io, output)
                end

            end
        end
    end
    return it_history, cost_history, sol
end

function eval_res(func, z) # res evaluation: ||z-v|| as in (5.1)
    hat_γ = 1 / sum(1 ./ func.γ) # gamma hat in Alg 2
    N = func.N
    temp = copy(z)
    temp .*= N / hat_γ
    for i = 1:N
        ∇f, ~ = gradient(func.F[i], z)
        temp .-= ∇f
    end
    temp .*= hat_γ / N

    v, ~ = CIAOAlgorithms.prox(func.g, temp, hat_γ) # v in Alg 2
    return norm(v .- z)
end

function loopnsave(iter, factor, Maxit, options) # the main iterations, by passing the iter to the 'for loop'
    R = eltype(x0) # element type  
    epoch = Int(Maxit * factor / N) # number of epochs 
    res_hist = Vector{R}(undef, 0)
    cost_hist = Vector{R}(undef, 0)
    it_hist = Vector{R}(undef, 0)
    cpu_hist = Vector{R}(undef, 0)
    it_sum = 0
    sol = copy(x0)

    func = options.func
    x_star = options.x_star
    if x_star !== nothing
        f_star = func(x_star)
    end
    resz = eval_res(func, x0)

    # initial point
    push!(cost_hist, func(x0))
    push!(res_hist, resz)
    push!(it_hist, 0)
    push!(cpu_hist, 0)

    println("init point res: ", resz); flush(stdout)
    println("grads per iteration is $(factor) and number of iterations needed is $(Maxit)"); flush(stdout)
    cnt = -1

    print_freq = 1 # frequency of the printing
    for stopwatch_state in take(iter, Maxit |> Int) # the main iterates of the optimizer 
        state = stopwatch_state[2]
        elapsed_time = stopwatch_state[1]*1e-9 # in sec
        it_sum += it_counter(iter.iter, state, options) # number of gradient evaluations (epoch = it_sum/N)
        
        if it_sum/N >= epoch + factor/N # as ls number in each iteration is not known 
            break
        end

        if cnt > 0 # cnt > 0 to skip the first iterates
            sol .= CIAOAlgorithms.solution(state)
            resz = eval_res(func, sol)
            cost = func(sol)

            push!(res_hist, resz)
            push!(cpu_hist, elapsed_time)
            push!(it_hist, it_sum/N) # epoch num 
            if x_star !== nothing
                push!(cost_hist, cost - f_star)  
            else
                push!(cost_hist, cost)  
            end
            
            if mod(cnt,print_freq) == 0
                gamma = isa(state.γ, R) ? state.γ : state.hat_γ
                # println("epoch: $(it_sum/N) - cost: $(cost) - γ: $(gamma) | γ max: $(maximum(state.γ)) - γ min: $(minimum(state.γ))"); flush(stdout)
                println("epoch: $(it_sum/N) - cost: $(cost) - res: $(resz) - cpu time: $(elapsed_time) | γ: $(gamma) - γ_max: $(maximum(state.γ)) - γ_min: $(minimum(state.γ))"); flush(stdout)
            end
        end
        cnt += 1
    end
    return it_hist, cost_hist, res_hist, cpu_hist, sol
end

struct StopwatchIterable{I} # for the cpu time calculations
    iter::I
end

function Base.iterate(iter::StopwatchIterable)
    next = Base.iterate(iter.iter)
    t0 = time_ns() # the stopwatch does not include the time for the initialization
    return dispatch(iter, t0, next)
end

function Base.iterate(iter::StopwatchIterable, (t0, state))
    next = Base.iterate(iter.iter, state)
    return dispatch(iter, t0, next)
end

function dispatch(iter::StopwatchIterable, t0, next)
    if next === nothing return nothing end
    return (time_ns()-t0, next[1]), (t0, next[2])
end

stopwatch(iter::I) where {I} = StopwatchIterable{I}(iter)

####################################### iteration counters #######################################
# it_counter specifies how many gradient evaluations the algorithm perfoms in each outer-loop iteration
it_counter(iter::CIAOAlgorithms.FINITO_basic_iterable, state::CIAOAlgorithms.FINITO_basic_state, options) = iter.N # it is =1 for each iter of basicFinito (Finito/MISO), but it is implemented in this way for speed!
it_counter(iter::CIAOAlgorithms.FINITO_LFinito_iterable, state::CIAOAlgorithms.FINITO_LFinito_state, options) = (options.epoch_per_iter - 1) * iter.N # -1 as it does not have lbfgs full update

function it_counter(iter::CIAOAlgorithms.FINITO_lbfgs_iterable, state::CIAOAlgorithms.FINITO_lbfgs_state, options)
    return (options.epoch_per_iter + round(log(options.β, CIAOAlgorithms.epoch_count(state)))) * iter.N # it considers ls backtracks on top of epoch_per_iter*N grad evaluations
end

function it_counter(iter::CIAOAlgorithms.FINITO_lbfgs_adaptive_iterable, state::CIAOAlgorithms.FINITO_lbfgs_adaptive_state, options)
    return (options.epoch_per_iter + state.ls_grad_eval + round(log(options.β, CIAOAlgorithms.epoch_count(state)))) * iter.N # it considers ls backtracks + ls 3 on top of epoch_per_iter * N grad evaluations
end

# third party solvers
it_counter(iter::CIAOAlgorithms.SGD_prox_iterable, state::CIAOAlgorithms.SGD_prox_state, options) = iter.N # it is =1 for each iter of SGD, but it is implemented in this way for speed!
it_counter(iter::CIAOAlgorithms.SAGA_basic_iterable, state::CIAOAlgorithms.SAGA_basic_state, options) = 1
it_counter(iter::CIAOAlgorithms.SAGA_prox_iterable, state::CIAOAlgorithms.SAGA_prox_state, options) = 2 * iter.N
it_counter(iter::CIAOAlgorithms.SVRG_prox_iterable, state::CIAOAlgorithms.SVRG_prox_state, options) = iter.N + iter.m
it_counter(iter::CIAOAlgorithms.SARAH_basic_iterable, state::CIAOAlgorithms.SARAH_basic_state, options) = iter.N + iter.m
it_counter(iter::CIAOAlgorithms.SARAH_prox_iterable, state::CIAOAlgorithms.SARAH_prox_state, options) = iter.N + iter.m
####################################### iteration counters #######################################