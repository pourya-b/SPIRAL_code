
function Comparisons_DNN!(
    stuff,
    x0,
    options;
) 
    func = options.func
    n = func.n
    N = func.N
    L = func.L

    # for saving and returning
    cost_history = Vector{Vector{T}}(undef, 0)
    res_history = Vector{Vector{T}}(undef, 0)
    it_history = Vector{Vector{T}}(undef, 0)

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
            DNN = stuff[i]["DNN"][j]
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
                    ls_tol = 1e-6,
                    γ = 0.999 * N / L,
                    DNN_training = DNN,
                )

                factor = N # number of grad evals per iteration (equal to N for default 'Finito/MISO')
                if LFinito # low memory with no ls
                    if DeepLFinito[1]
                        factor = N * (1 + DeepLFinito[3])
                    else
                        factor = (options.epoch_per_iter - 1) * N
                    end
                elseif lbfgs
                    if DeepLFinito[1]
                        factor = (2+  DeepLFinito[3]) * N
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
                    N = N,
                    data = data,
                    DNN_config = DNN_config!,
                )

                it_hist, cost_hist, res_hist, sol =
                    loopnsave_dnn(iter, factor, Maxit, options, L) # the main iterations using the iterable 'iter'. func.L is used for the res evaluation

                # for returning
                push!(cost_history, cost_hist)
                push!(res_history, res_hist)
                push!(it_history, it_hist)

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
                        Labelsweep[sweeping], # for j
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
                        Labelsweep[sweeping],
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

function eval_gradient(γ) # evaluating res = |z-prox(z)| for a given γ
    isa(γ, T) && (γ = fill(γ, (N,))) 
    hat_γ = 1 / sum(1 ./ γ)

    z = DNN_config!() # current parameters of DNN
    gr = gradient(ps) do # an object of Zygote. Gradient calculation of the smooth part (cost function) wrt the dnn parameters in the current point (z) 
        NewLoss(batchmemaybe(data)...)
    end
    ∇f_N = DNN_config!(gs=gr) # Gradient vector returned by the function. (the same as gr, but in the vector form), already divided by N as the cost function is divided by N
    av = z - hat_γ * ∇f_N
    v, ~ = prox(g, av, hat_γ)
    return norm(z - v) 
end

function loopnsave_dnn(iter, factor, Maxit, options, L_cst) # the main iterations, by passing the iter to the 'for loop'
    epoch = Int(Maxit * factor / N) # number of epochs 
    res_hist = Vector{T}(undef, 0)
    cost_hist = Vector{T}(undef, 0)
    it_hist = Vector{T}(undef, 0)
    it_sum = 0
    sol = copy(w0)

    func = options.func
    resz = eval_gradient(1/L_cst)

    # initial point
    push!(cost_hist, func(data))
    push!(res_hist, resz)
    push!(it_hist, 0)

    println("init point res: ", resz); flush(stdout)
    println("grads per iteration is $(factor) and number of iterations needed is $(Maxit)"); flush(stdout)
    cnt = -1

    print_freq = 1 # frequency of the printing

    for state in take(iter, Maxit |> Int) # the main iterates of the optimizer 
        it_sum += it_counter(iter, state, options) # number of gradient evaluations (epoch = it_sum/N)
        
        if it_sum/N >= epoch + factor/N # as ls number in each iteration is not known 
            break
        end

        if cnt > 0 # cnt > 0 to skip the first iterates
            sol .= CIAOAlgorithms.solution(state)
            resz = eval_gradient(1/L_cst)
            cost = func(data)

            push!(res_hist, resz)
            push!(it_hist, it_sum/N) # epoch num 
            push!(cost_hist, cost)  
            
            if mod(cnt,print_freq) == 0
                gamma = isa(state.γ, T) ? state.γ : state.hat_γ
                # println("epoch: $(it_sum/N) - cost: $(cost) - γ: $(gamma) | γ max: $(maximum(state.γ)) - γ min: $(minimum(state.γ))"); flush(stdout)
                println("epoch: $(it_sum/N) - cost: $(cost) - res: $(resz) | γ: $(gamma)"); flush(stdout)
            end
        end
        cnt += 1
    end
    return it_hist, cost_hist, res_hist, sol
end

batchmemaybe(x) = tuple(x)
batchmemaybe(x::Tuple) = x

function eval_DNN(str = nothing) # evaluates the DNN with counting the number of errors in the class prediction
    error_cnt = 0
    for i in 1:N
        if (argmax(softmax(DNN_model(x_train[i,:]))) != argmax(y_train[i,:]))
            error_cnt += 1
        end
    end
    println(string(str, ": Number of errors is $(error_cnt) out of $(N) samples\n")); flush(stdout)
end

function return_model_params(model) # return a copy of model parameters: ds = copy(ps), where ps is the model parameters as a Zygote object
    ps = params(model)
    ps_len = length(ps)
    struc = []
    for j in 1:2:ps_len-2
        push!(struc,Dense(size(ps[j],2),size(ps[j],1),tanh))
    end
    j = ps_len-1
    push!(struc,Dense(size(ps[j],2),size(ps[j],1)))

    dummy = Chain(struc) 
    ds = params(dummy)
    for i in 1:ps_len
        ds[i] .= ps[i]
    end
    return ds
end

function DNN_config!(;gs=nothing) # in: nothing out-> vectorized DNN params/ in: gs=grads out -> vectorized grad/ in:gs=params out -> nothing [sets DNN params by gs]
    inx = 1
    ps_len = length(ps)
    config_length = 0
    for j in 1:2:ps_len
        config_length += (size(ps[j],2)+1) * size(ps[j],1)
    end

    if (typeof(gs) == Zygote.Grads)
        out = zeros(config_length)
        for par in ps
            xp = gs[par]
            xpr = reshape(xp,(length(xp),1))
            Len = length(xpr)
            out[inx:inx+Len-1] = xpr
            inx += Len
        end
        return out
    elseif gs==nothing
        out = zeros(config_length)
        for par in ps
            xpr = reshape(par,(length(par),1))
            Len = length(xpr)
            out[inx:inx+Len-1] = xpr
            inx += Len
        end
        return out
    else
        for i=1:ps_len
            Len = length(ps[i])
            ps[i] .= reshape(gs[inx:inx+Len-1],size(ps[i]))
            inx += Len
        end
        return nothing
    end
end


####################################### iteration counters #######################################
function it_counter(iter::CIAOAlgorithms.FINITO_lbfgs_iterable_DNN, state::CIAOAlgorithms.FINITO_lbfgs_state_DNN, options)
    return (options.epoch_per_iter + round(log(options.β, CIAOAlgorithms.epoch_count(state)))) * iter.N # it considers ls backtracks on top of epoch_per_iter*N grad evaluations
end

# third party solvers
it_counter(iter::CIAOAlgorithms.SARAH_prox_DNN_iterable, state::CIAOAlgorithms.SARAH_prox_DNN_state, options) = iter.N + iter.m
it_counter(iter::CIAOAlgorithms.SGD_prox_DNN_iterable, state::CIAOAlgorithms.SGD_prox_DNN_state, options) = iter.N # it is =1 for each iter of SGD, but it is implemented in this way for speed!
####################################### iteration counters #######################################