# utility function for SGD, it runs SGD according to the selected options below in func, options.

function Comparisons_sgd!(
    x0,
    options;
    label = "SGD",
    diminishing = true,    # true for diminishing stepsize, false for constant stepsize = γ, provided above
    η0 = 0.1,              # for the stepsize: η0/(1+ η_tilde * epoch_counter)
    η_tilde = 0.5,
) 
    func = options.func
    n = func.n
    N = func.N
    L = func.L
    m = func.m
    DNN_flag = func.DNN
    L_ratio = Int(floor(maximum(L)/minimum(L)))
    
    println("\n\nusing $(label) ...")

    solver = CIAOAlgorithms.SGD{R}(diminishing=diminishing, η0=η0, η_tilde=η_tilde, DNN = DNN_flag)  # returns the SGD solver based on the parameters
    iter = CIAOAlgorithms.iterator(solver, x0, F = func.F, g = func.g, L = func.L, N = N, data=func.data, DNN_config=func.configure_func) # iterable handle to pass to the for loop
    iter = stopwatch(iter)

    factor = N # number of grad evals per outer iteration (SGD is implemented in this way for speed, the factor is actually = batch size = 1)
    Maxit = Int(ceil(epoch * N / factor)) # outer iteration number needed for the algorithm

    if DNN_flag # if the cost function is a DNN or not
        it_hist, cost_hist, res_hist, sol =
                    loopnsave_dnn(iter, factor, Maxit, options, func.L) # the main iterations using the iterable 'iter'. # In utilities_DNN.jl
    else
        it_hist, cost_hist, res_hist, cpu_hist, sol =
                    loopnsave(iter, factor, Maxit, options) # the main iterations using the iterable 'iter'. # In utilities.jl
    end

    # saving 
    output = [it_hist cost_hist]
    mkpath(string("plot_data/",str,"/SGD/cost"))
    open(
        string(
            "plot_data/",
            str,
            "SGD/cost/",
            "N_",
            N,
            "_n_",
            n,
            "_Lratio_",
            L_ratio,
            "_",
            label,
            "_diminishing_",
            diminishing,
            "_eta_",
            η0,
            "_eta_tilde_",
            η_tilde,
            ".txt",
        ),
      "w",
    ) do io
        writedlm(io, output)
    end

    output = [it_hist res_hist]
    mkpath(string("plot_data/",str,"/SGD/res"))
    open(
        string(
        "plot_data/",
        str,
        "SGD/res/",
        "N_",
        N,
        "_n_",
        n,
        "_Lratio_",
        L_ratio,
        "_",
        label,
        "_diminishing_",
        diminishing,
        "_eta_",
        η0,
        "_eta_tilde_",
        η_tilde,
        ".txt",
    ),
    "w",
    ) do io
        writedlm(io, output)
    end

    output = [cpu_hist res_hist]               
    mkpath(string("plot_data/",str,"/SGD/cpu"))
    open(
        string(
        "plot_data/",
        str,
        "SGD/cpu/",
        "N_",
        N,
        "_n_",
        n,
        "_Lratio_",
        L_ratio,
        "_",
        label,
        "_diminishing_",
        diminishing,
        "_eta_",
        η0,
        "_eta_tilde_",
        η_tilde,
        ".txt",
    ),
    "w",
    ) do io
        writedlm(io, output)
    end
end  