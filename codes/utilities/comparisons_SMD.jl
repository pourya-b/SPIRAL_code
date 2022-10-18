# utility function for SMD, it runs the algorithm according to the selected options below in the func, options.

function Comparisons_smd!(
    x0,
    options;
    label = "SMD",
    diminishing = true,
) 
        func = options.func
        n = func.n
        N = func.N
        L = func.L
        γ = func.γ
    
        L_M      = maximum(L)
        L_ratio = Int(floor(maximum(L)/minimum(L)))

    println("\n\nusing $(label) ...") 
    
    solver = BregmanBC.SMD{R}(diminishing = diminishing)
    iter = BregmanBC.iterator(solver, x0, F = func.F, g = func.g, H = func.H, L = L_M, N = N)

    factor = N                          # number of grad evals per outer iteration (it is implemented in this way for speed)
    Maxit = floor(N * epoch / factor)   # outer iteration number needed for the algorithm
    iter = stopwatch(iter)

    dist_hist, it_hist, z_sol, dist2_hist, cost_hist, cpu_hist  = getresidue!(iter, factor, Maxit, γ, func.H, options)   # the main iterations using the iterable 'iter'. # In utilities_breg.jl
    
    save_str = "N_$(N)_n_$(n)_p_$(image_num)_$(label)_Lratio_$(L_ratio)_lambda_$(round(λ * N, digits = 5))_pfail_$(p_fail)"

    #### ------------------------------------ solution visualization ---------------------------------
    Gray.(reshape(z_sol, 16,16))
    z_clamp = map(clamp01nan, z_sol)   # bring back to 0-1 range!
    z_clamp_ = map(clamp01nan, -z_sol)   # bring back to 0-1 range!
    save(string("solutions/",digits,"/",save_str,".png"), colorview(Gray, reshape(abs.(z_clamp), 16, 16)'))
    save(string("solutions/",digits,"/",save_str,"_.png"), colorview(Gray, reshape(abs.(z_clamp_), 16, 16)'))
    #### ---------------------------------------------------------------------------------------------

    # for saving 
    output = [it_hist dist_hist]
    mkpath(string("plot_data/",str,"/SMD/zzplus"))
    open(
        string(
            "plot_data/",
            str,
            "/SMD/zzplus/",
            save_str,
            ".txt",
        ),
        "w",
    ) do io
        writedlm(io, output)
    end

    output = [it_hist dist2_hist]
    mkpath(string("plot_data/",str,"/SMD/Hzzplus"))
    open(
        string(
            "plot_data/",
            str,
            "/SMD/Hzzplus/",
            save_str,
            ".txt",
        ),
        "w",
    ) do io
        writedlm(io, output)
    end

    output = [it_hist cost_hist]
    mkpath(string("plot_data/",str,"/SMD/cost"))
    open(
        string(
            "plot_data/",
            str,
            "/SMD/cost/",
            save_str,
            ".txt",
        ),
        "w",
    ) do io
        writedlm(io, output)
    end

    output = [cpu_hist dist_hist]
    mkpath(string("plot_data/",str,"/SMD/cpu"))
    open(
        string(
            "plot_data/",
            str,
            "/SMD/cpu/",
            save_str,
            ".txt",
        ),
        "w",
    ) do io
        writedlm(io, output)
    end
end
