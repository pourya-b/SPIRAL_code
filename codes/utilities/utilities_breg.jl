# utility for Bregman SPIRAL. It compares different versions of the Bregman SPIRAL according to the stuff list in Comparisons_breg function. Also, it compares different versions of the SPIRAL according to the stuff list in Comparisons_eucl function.
# function call map: Comparisons_breg (or Comparisons_eucl) --> getresidue --> Residue_z 

function Comparisons_spiral_breg!(
    stuff,
    x0,
    options,
) 
    func = options.func
    n = func.n
    N = func.N
    L = func.L

    # for saving and returning
    dist_history = Vector{Vector{R}}(undef, 0)
    it_history = Vector{Vector{R}}(undef, 0)
    cpu_history = Vector{Vector{R}}(undef, 0)

    cnt = 0 # counter
    t1 = length(stuff)
    sol = copy(x0)
    for i = 1:t1 # loop for all the matrials in stuff
        t2 = length(stuff[i]["sweeping"])

        for j = 1:t2 # loop over index update strategy between ["clc", "sfld"]
            LBFinito = stuff[i]["LFinito"]
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

                solver = BregmanBC.Bregman_Finito{R}( # returns the suitable solver (SPIRAL bregman version) based on the parameters
                    sweeping = sweeping,
                    LBFinito = LBFinito,
                    minibatch = minibatch,
                    lbfgs = lbfgs,
                    adaptive = adaptive,
                    β = options.β,
                    ls_tol = 1e-6,
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
                if LBFinito # SPIRAL-no-ls
                    factor = (options.epoch_per_iter - 1) * N
                elseif lbfgs
                    factor = (options.epoch_per_iter) * N
                end

                Maxit = Int(ceil(epoch * N/ factor)) # outer iteration number needed for the algorithm
                iter = BregmanBC.iterator(solver, x0, F = func.F, g = func.g, H = H, L = L_s, N = N)
                iter = stopwatch(iter)
                dist_hist, it_hist, z_sol, dist2_hist, cost_hist, cpu_hist  = getresidue!(iter, factor, Maxit, γ, H, options) # the main iterations using the iterable 'iter'.

                save_str = "N_$(N)_n_$(n)_p_$(image_num)_batch_$(size_batch)_$(stuff[i]["label"])_$(Labelsweep[sweeping])_Lratio_$(L_ratio)_lambda_$(round(λ * N, digits = 5))_pfail_$(p_fail)"

                #### ------------------------------------ solution visualization ---------------------------------
                Gray.(reshape(z_sol, 16,16))
                z_clamp = map(clamp01nan, z_sol)   # bring back to 0-1 range!
                z_clamp_ = map(clamp01nan, -z_sol)   # bring back to 0-1 range!
                save(string("solutions/",digits,"/",save_str,".png"), colorview(Gray, reshape(abs.(z_clamp), 16, 16)'))
                save(string("solutions/",digits,"/",save_str,"_.png"), colorview(Gray, reshape(abs.(z_clamp_), 16, 16)'))
                #### ---------------------------------------------------------------------------------------------

                # for returning
                push!(dist_history, dist_hist)
                push!(it_history, it_hist)
                push!(cpu_history, cpu_hist)

                # for saving 
                output = [it_hist dist_hist]
                mkpath(string("plot_data/",str,"/zzplus"))
                open(
                    string(
                        "plot_data/",
                        str,
                        "/zzplus/",
                        save_str,
                        ".txt",
                    ),
                    "w",
                ) do io
                    writedlm(io, output)
                end

                output = [it_hist dist2_hist]
                mkpath(string("plot_data/",str,"/Hzzplus"))
                open(
                    string(
                        "plot_data/",
                        str,
                        "/Hzzplus/",
                        save_str,
                        ".txt",
                    ),
                    "w",
                ) do io
                    writedlm(io, output)
                end

                output = [it_hist cost_hist]
                mkpath(string("plot_data/",str,"/cost"))
                open(
                    string(
                        "plot_data/",
                        str,
                        "/cost/",
                        save_str,
                        ".txt",
                    ),
                    "w",
                ) do io
                    writedlm(io, output)
                end

                output = [cpu_hist dist_hist]
                mkpath(string("plot_data/",str,"/cpu"))
                open(
                    string(
                        "plot_data/",
                        str,
                        "/cpu/",
                        save_str,
                        ".txt",
                    ),
                    "w",
                ) do io
                    writedlm(io, output)
                end
            end
        end
    end
    return dist_history, it_history
end

function Comparisons_spiral_eucl!(
    stuff,
    x0,
    options,
) 
    func = options.func
    n = func.n
    N = func.N
    L = func.L

    # for saving and returning
    dist_history = Vector{Vector{R}}(undef, 0)
    it_history = Vector{Vector{R}}(undef, 0)
    cpu_history = Vector{Vector{R}}(undef, 0)

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

                solver = CIAOAlgorithms.Finito{R}( # returns the suitable solver (SPIRAL version) based on the parameters
                    sweeping = sweeping,
                    LFinito = LFinito,
                    DeepLFinito = DeepLFinito,
                    minibatch = minibatch,
                    lbfgs = lbfgs,
                    adaptive = adaptive,
                    β = options.β,
                    ls_tol = 1e-6,
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
                    L = L_s,
                    N = N,
                )

                iter = stopwatch(iter)
                dist_hist, it_hist, z_sol, dist2_hist, cost_hist, cpu_hist  = getresidue!(iter, factor, Maxit, γ, H, options)
    
                save_str = "N_$(N)_n_$(n)_p_$(image_num)_batch_$(size_batch)_$(stuff[i]["label"])_$(Labelsweep[sweeping])_Lratio_$(L_ratio)_lambda_$(round(λ * N, digits = 5))_pfail_$(p_fail)"

                #### ------------------------------------ solution visualization ---------------------------------
                Gray.(reshape(z_sol, 16,16))
                z_clamp = map(clamp01nan, z_sol)   # bring back to 0-1 range!
                z_clamp_ = map(clamp01nan, -z_sol)   # bring back to 0-1 range!
                save(string("solutions/",digits,"/",save_str,".png"), colorview(Gray, reshape(abs.(z_clamp), 16, 16)'))
                save(string("solutions/",digits,"/",save_str,"_.png"), colorview(Gray, reshape(abs.(z_clamp_), 16, 16)'))
                #### ---------------------------------------------------------------------------------------------

                # for returning
                push!(dist_history, dist_hist)
                push!(it_history, it_hist)
                push!(cpu_history, cpu_hist)

                # for saving 
                output = [it_hist dist_hist]
                mkpath(string("plot_data/",str,"/zzplus"))
                open(
                    string(
                        "plot_data/",
                        str,
                        "/zzplus/",
                        save_str,
                        ".txt",
                    ),
                    "w",
                ) do io
                    writedlm(io, output)
                end

                output = [it_hist dist2_hist]
                mkpath(string("plot_data/",str,"/Hzzplus"))
                open(
                    string(
                        "plot_data/",
                        str,
                        "/Hzzplus/",
                        save_str,
                        ".txt",
                    ),
                    "w",
                ) do io
                    writedlm(io, output)
                end

                output = [it_hist cost_hist]
                mkpath(string("plot_data/",str,"/cost"))
                open(
                    string(
                        "plot_data/",
                        str,
                        "/cost/",
                        save_str,
                        ".txt",
                    ),
                    "w",
                ) do io
                    writedlm(io, output)
                end

                output = [cpu_hist dist_hist] # could be dist2_hist
                mkpath(string("plot_data/",str,"/cpu"))
                open(
                    string(
                        "plot_data/",
                        str,
                        "/cpu/",
                        save_str,
                        ".txt",
                    ),
                    "w",
                ) do io
                    writedlm(io, output)
                end
            end
        end
    end
    return dist_history, it_history
end

function getresidue!(iter, factor, Maxit, γ::Array{Float64}, H, options) # iter now is stopwatch struct

    dist_hist = Vector{R}(undef, 0)
    dist2_hist = Vector{R}(undef, 0)
    cost_hist = Vector{R}(undef, 0)
    it_hist = Vector{R}(undef, 0)
    cpu_hist = Vector{R}(undef, 0)
    sol_last = copy(x0)

    # initial residue
    cost = iter.iter.g(x0)
    for i = 1:N
        cost += iter.iter.F[i](x0) / N
    end
    D1, D2 = Residue_z!(H, iter.iter.F, iter.iter.g, x0, γ)
    push!(dist_hist, D1)
    push!(dist2_hist, D2)
    push!(cost_hist, cost)
    push!(it_hist, 0)
    push!(cpu_hist, 0)
    
    it_sum = 0
    cnt = -1

    for stopwatch_state in take(iter, Maxit + 2 |> Int) # the main iterates of the optimizer 
        state = stopwatch_state[2]
        elapsed_time = stopwatch_state[1]*1e-9 # in sec
        it_sum += it_counter(iter.iter, state, options) # number of gradient evaluations (epoch = it_sum/N)
        
        if it_sum/N >= epoch + factor/N # as ls number in each iteration is not known 
            break
        end
        
        if cnt > 0   # cnt > 0 to skip the first iterates       
            z = solution(state)
            # println("epoch: $(it_sum/N) - output: $(norm(z))") 


            cost = iter.iter.g(z)
            for i = 1:N
                cost += iter.iter.F[i](z) / N
            end

            D1, D2 = Residue_z!(H, iter.iter.F, iter.iter.g, z, γ)

            push!(dist_hist, D1)        # |z - prox(z)|
            push!(dist2_hist, D2)       # nabla H(z) - nabla H(v) for v in prox(z)
            push!(cost_hist, cost)      # cost
            push!(it_hist, it_sum/N)    # indicating number of epochs
            push!(cpu_hist, elapsed_time)

            println("epoch: $(it_sum/N) - cost: $(cost) - res_1: $(D1) - res_2: $(D2) - cpu time: $(elapsed_time) - sol: $(norm(z))") 
            sol_last .= z 
        end
        cnt += 1
    end
    return dist_hist, it_hist, sol_last, dist2_hist, cost_hist, cpu_hist
end


function Residue_z!(H, F, g, z, γ::Array{Float64})
    N = length(F)
    NHZ = computeHatH(H, F, z, γ, N)  # nabla hat H(z) 
    v, ~ = prox_Breg(H, g, NHZ, γ) # v= z^+
    NHZ2 = computeHatH(H, F, v, γ, N)  # nabla hat H(z) 

    return norm( v .- z ), norm(NHZ2 .- NHZ)
end


function computeHatH(H::Array{Th}, F, x, γ::Array{Float64}, N) where {Th}
    temp = zero(x)
    for i = 1:N
        ∇h, ~ = BregmanBC.gradient(H[i], x)
        temp .+= ∇h ./ γ[i]
        ∇f, ~ = BregmanBC.gradient(F[i], x)
        temp .-= ∇f ./ N
    end
    return temp
end

function initializeX(A, b, N) # different methods of initializations
    ################ Duchi alg2.....see Davis 2018 Thoerem 3.8 also
    N = size(A,1)
    idx = findall(x -> x <= mean(b), b)
    X = zeros(n,n)
    for i in idx
      a_i = A[i,:]
      X .+=  a_i * a_i' 
    end
    en = eigen(X)
    d = en.vectors[:, 1]
    x0 = sqrt(mean(b))*d
    
    ################ initial point according to Wang!
    # temp = Vector{R}(undef, 0)
    # for i = 1:N
    #     push!(temp, b[i] / norm(A[i, :])^2)
    # end
    # quant56 = quantile(temp, 5 / 6)

    # idx = findall(x -> x >= quant56, temp)
    # X = zeros(n, n)
    # for i in idx
    #     a_i = A[i, :]
    #     X .+= a_i * a_i' ./ norm(a_i)^2
    # end
    # en = eigen(X)
    # d = en.vectors[:, end]
    # x0 = sqrt(sum(b) / N) * d

    ################ initial point according to Zhang!
    # lam =  sqrt( quantile(b, 1/2) / 0.455 )  
    # idx = findall(x -> abs.(x) <= 9 * lam, b)    

    # X = zeros(n, n)
    # for i in idx
    #     X .+= b[i] * A[i, :] * A[i, :]'
    # end
    # X ./= N 
    # en = eigen(X)
    # d = en.vectors[:, end]
    # x0 = lam * d
    # x0 ./= maximum(x0) ########### I added!     
    return x0
end


function Ab_image(x_star,k, p_fail)
    n = length(x_star)
    N = k * n

    # Hadamand matrices
    Hdmd = hadamard(n) ./ sqrt(n)
    HS = Vector{Array}(undef, 0)
    for i = 1:k
        # S = diagm(0 => rand([1, -1], n))
        S = diagm(0 => sign.(randn(n)))
        # println(S[1:3,1:3])
        push!(HS,  Hdmd * S)
        # push!(HS,  S * Hdmd )
    end
    A = vcat(HS...)

    # generating b
    b = (A * x_star) .^ 2
    # curropt with probability p_fail
    for i in eachindex(b)
        # println("b_i")
        if rand() < p_fail
            b[i] = 0
            # println("b_i zero")
        end
    end
    return A, b
end 

function Ab_rand(x_star,k, p_fail)
    n = length(x_star)
    N = k * n

    A = randn(N, n)
    # generating b
    b = (A * x_star) .^ 2
    # curropt with probability p_fail
    for i in eachindex(b)
        if rand() < p_fail
            b[i] = 0
        end
    end
    return A, b
end 


####################################### iteration counters #######################################
# it_counter specifies how many gradient evaluations the algorithm perfoms in each outer-loop iteration
it_counter(iter::BregmanBC.Breg_FINITO_basic_iterable, state::BregmanBC.Breg_FINITO_basic_state, options) = iter.N # it is =1 for each iter of basicFinito (Finito/MISO), but it is implemented in this way for speed!
it_counter(iter::BregmanBC.LBreg_Finito_iterable, state::BregmanBC.LBreg_Finito_state, options) = (options.epoch_per_iter - 1) * iter.N # -1 as it does not have lbfgs full update
function it_counter(iter::BregmanBC.LBreg_Finito_lbfgs_iterable, state::BregmanBC.LBreg_Finito_lbfgs_state, options)
    return (options.epoch_per_iter + round(log(options.β, BregmanBC.epoch_count(state)))) * iter.N # it considers ls backtracks on top of epoch_per_iter*N grad evaluations
end

function it_counter(iter::BregmanBC.LBreg_Finito_lbfgs_adaptive_iterable, state::BregmanBC.LBreg_Finito_lbfgs_adaptive_state, options)
    return (options.epoch_per_iter + state.ls_grad_eval + round(log(options.β, BregmanBC.epoch_count(state)))) * iter.N # it considers ls backtracks on top of epoch_per_iter*N grad evaluations
end

# third party solvers
it_counter(iter::BregmanBC.SMD_basic_iterable, state::BregmanBC.SMD_basic_state, options) = iter.N # it is =1 for each iter of SMD, but it is implemented in this way for speed!
####################################### iteration counters #######################################