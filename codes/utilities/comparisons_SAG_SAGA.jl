# utility function for SAG-SAGA, it runs the algorithm according to the selected options below in the func, options.

function Comparisons_saga!(
    x0,
    options;
    label = "SAGA",
    prox = true,
) 
        func = options.func
        n = func.n
        N = func.N
        L = func.L
        DNN_flag = func.DNN
        convex_flag = func.convex
        prox_flag = prox
    
        L_M      = maximum(L)
        L_ratio = Int(floor(maximum(L)/minimum(L)))

        println("\n\nsolving using $(label) ...")

        if convex_flag
            γ_SAGA = 1 / (3 * L_M) # according to supp. material of "SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives"
        else
            γ_SAGA = 1 / (5 * N * L_M)  # according to thm 3 of "Proximal Stochastic Methods for Nonsmooth Nonconvex Finite-Sum Optimization", for batch size = 1
        end

        solver = CIAOAlgorithms.SAGA{R}(γ = γ_SAGA, prox_flag = prox_flag)
        iter   = CIAOAlgorithms.iterator(solver, x0, F = func.F, g = func.g, N = N)
        iter = stopwatch(iter)

        factor = 2 * N # according to Alg 2 of "Proximal Stochastic Methods for Nonsmooth Nonconvex Finite-Sum Optimization", with the equal sets J = I and batch size = 1. In each outer iteration there are two gradient evaluations. It is implemented for N outer iterations for speed.
        Maxit = Int(ceil(epoch * N / factor)) # outer iteration number needed for the algorithm

        it_hist, cost_hist, res_hist, cpu_hist, sol =
                        loopnsave(iter, factor, Maxit, options) # the main iterations using the iterable 'iter'. # In utilities.jl
        
        # saving 
        output = [it_hist cost_hist]
        mkpath(string("plot_data/",str,label,"/cost"))
        open(
            string(
                "plot_data/",
                str,
                label,
                "/cost/",
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
        mkpath(string("plot_data/",str,label,"/res"))
        open(
            string(
            "plot_data/",
            str,
            label,
            "/res/",
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
        mkpath(string("plot_data/",str,label,"/cpu"))
        open(
            string(
            "plot_data/",
            str,
            label,
            "/cpu/",
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
