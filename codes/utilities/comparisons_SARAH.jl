# utility function for SARAH, it runs SARAH according to the selected options below in func, options.

function Comparisons_sarah!(
  x0,
  options;
  label = "SARAH",
) 
    func = options.func
    n = func.n
    N = func.N
    L = func.L
    m = func.m
    DNN_flag = func.DNN

    if L === nothing
      @warn "smoothness constant is absent"
      return nothing
    else
      println("new stepsize")
      L_M      = sqrt(sum(L.^2)/N) # based on (4) in "ProxSARAH: An Efficient Algorithmic Framework for Stochastic Composite Nonconvex Optimization"
      # L_M = maximum(L) # wrong
      L_ratio = Int(floor(maximum(L)/minimum(L)))
    end
    println("\n\nusing $(label) ...")  

    batch = 1
    ω = (3*(N-batch))/(2*batch*(N-1))
    γ_sarah = (2*sqrt(ω * m)) / (4*sqrt(ω * m) + 1) # based on thm 8 of "ProxSARAH: An Efficient Algorithmic Framework for Stochastic Composite Nonconvex Optimization"
    ꞵ = 1/(L_M * sqrt(ω * m)) # based on thm 8 of "ProxSARAH: An Efficient Algorithmic Framework for Stochastic Composite Nonconvex Optimization"
    
    solver = CIAOAlgorithms.SARAH{R}(γ = γ_sarah, ꞵ = ꞵ, m = m, DNN = DNN_flag)    # returns the SARAH solver based on the parameters
    iter = CIAOAlgorithms.iterator(solver, x0, F = func.F, g = func.g, N = func.N, data=func.data, DNN_config=func.configure_func)  # iterable handle to pass to the for loop
    iter = stopwatch(iter)
    
    factor = N + m                        # number of grad evals per outer iteration
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
    mkpath(string("plot_data/",str,"/SARAH/cost"))
    open(
        string(
            "plot_data/",
            str,
            "SARAH/cost/",
            "N_",
            N,
            "_n_",
            n,
            "_Lratio_",
            L_ratio,
            "_",
            label,
            ".txt",
        ),
      "w",
    ) do io
        writedlm(io, output)
    end

    output = [it_hist res_hist]
    mkpath(string("plot_data/",str,"/SARAH/res"))
    open(
        string(
          "plot_data/",
          str,
          "SARAH/res/",
          "N_",
          N,
          "_n_",
          n,
          "_Lratio_",
          L_ratio,
          "_",
          label,
          ".txt",
      ),
        "w",
    ) do io
        writedlm(io, output)
    end

    output = [cpu_hist res_hist]
    mkpath(string("plot_data/",str,"/SARAH/cpu"))
    open(
        string(
          "plot_data/",
          str,
          "SARAH/cpu/",
          "N_",
          N,
          "_n_",
          n,
          "_Lratio_",
          L_ratio,
          "_",
          label,
          ".txt",
      ),
        "w",
    ) do io
        writedlm(io, output)
    end
  end 
