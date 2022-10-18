# utility function for SVRG, it runs SVRG according to the selected options below in the func, options.

function Comparisons_svrg!(
  x0,
  options;
  label = "SVRG",
  alg = "Reddi",        # choose options in {"Reddi", "SVRG+"} according which the stepsize is calculated
) 
    func = options.func
    n = func.n
    N = func.N
    L = func.L
    m = func.m          # number of iterations in the inner loop  
    DNN_flag = func.DNN # for DNN training

    if L === nothing
      @warn "smoothness constant is absent"
      return nothing
    else
      L_M      = maximum(L)
      L_ratio = Int(floor(maximum(L)/minimum(L)))
    end

    println("\n\nusing $(label) with stepsize according to $(alg) ...")  

    if alg == "Reddi"     # according to thm 1 of "Proximal Stochastic Methods for Nonsmooth Nonconvex Finite-Sum Optimization", for batch size = 1
      γ_svrg = 1/(3 * N * L_M)
      @assert m == N
    elseif alg == "SVRG+" # according to thm 3 of "A Simple Proximal Stochastic Gradient Method for Nonsmooth Nonconvex Optimization", for m \neq \sqrt(batch size)
      γ_svrg = 1/(6 * m * L_M)
    end

    # γ_svrg = 1/(7*maximum(L))

    solver = CIAOAlgorithms.SVRG{R}(γ = γ_svrg, m = m, DNN = DNN_flag) # returns the SVRG solver based on the parameters
    iter = CIAOAlgorithms.iterator(solver, x0, F = func.F, g = func.g, N = N)    # iterable handle to pass to the for loop
    iter = stopwatch(iter)

    factor = N + m                        # number of grad evals per outer iteration
    Maxit = Int(ceil(epoch * N / factor)) # outer iteration number needed for the algorithm

    it_hist, cost_hist, res_hist, cpu_hist, sol =
                    loopnsave(iter, factor, Maxit, options) # the main iterations using the iterable 'iter'. # In utilities.jl
    
    # saving 
    output = [it_hist cost_hist]
    mkpath(string("plot_data/",str,"/SVRG/cost"))
    open(
        string(
            "plot_data/",
            str,
            "SVRG/cost/",
            "N_",
            N,
            "_n_",
            n,
            "_Lratio_",
            L_ratio,
            "_",
            label,
            "_",
            "alg",
            "_",
            alg,
            ".txt",
        ),
      "w",
    ) do io
        writedlm(io, output)
    end

    output = [it_hist res_hist]
    mkpath(string("plot_data/",str,"/SVRG/res"))
    open(
        string(
          "plot_data/",
          str,
          "SVRG/res/",
          "N_",
          N,
          "_n_",
          n,
          "_Lratio_",
          L_ratio,
          "_",
          label,
          "_",
          "alg",
          "_",
          alg,
          ".txt",
      ),
        "w",
    ) do io
        writedlm(io, output)
    end

    output = [cpu_hist res_hist]
    mkpath(string("plot_data/",str,"/SVRG/cpu"))
    open(
        string(
          "plot_data/",
          str,
          "SVRG/cpu/",
          "N_",
          N,
          "_n_",
          n,
          "_Lratio_",
          L_ratio,
          "_",
          label,
          "_",
          "alg",
          "_",
          alg,
          ".txt",
      ),
        "w",
    ) do io
        writedlm(io, output)
    end
end 