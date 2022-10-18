# Bregman Finito/MISO

struct Breg_FINITO_basic_iterable{R<:Real,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Tg}
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    H::Any                  # distance generating function 
    x0::Tx                  # initial point
    N::Int                  # number of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i	
    γ::Maybe{Union{Array{R},R}}  # stepsizes 
    sweeping::Int8          # to only use one stepsize γ
    batch::Int              # batch size
    α::R                    # in (0, 1), e.g.: 0.99
end

mutable struct Breg_FINITO_basic_state{R<:Real,Tx,Mx}
    s::Mx            # table of x_j- γ_j/N nabla f_j(x_j) 
    γ::Array{R}             # stepsize parameters 
    av::Tx                  # the running average
    z::Tx
    ind::Array{Array{Int}}  # running index set 
    d::Int                  # number of batches
    epoch_cnt::R            # epoch counter
    
    # some extra placeholders  
    ∇f_temp::Tx             # placeholder for gradients 
    ∇h_temp::Tx             # placeholder for gradients 
    idxr::Int               # running idx in the iterate 
    idx::Int                # location of idxr in 1:N 
    inds::Array{Int}        # needed for shuffled only  
end

function Breg_FINITO_basic_state(s::Mx, γ::Array{R}, av::Tx, z::Tx, ind, d) where {R,Tx,Mx}
    return Breg_FINITO_basic_state{R,Tx,Mx}(
        s,
        γ,
        av,
        z,
        ind,
        d,
        0.0,
        zero(av),
        zero(av),
        Int(1),
        Int(0),
        collect(1:d),
    )
end

function Base.iterate(iter::Breg_FINITO_basic_iterable{R,C,Tx}) where {R,C,Tx}
    N = iter.N
    n = size(iter.x0)[1]
    # define the batches
    r = iter.batch # batch size 
    if iter.sweeping == 1 # cyclic
        ind = [collect(1:r)] # placeholder
    else # shuffled
        ind = Vector{Vector{Int}}(undef, 0)
        d = Int(floor(N / r))
        for i = 1:d
            push!(ind, collect(r*(i-1)+1:i*r))
        end
        r * d < N && push!(ind, collect(r*d+1:N))
    end
    d = cld(N, r) # number of batches  

    # updating the stepsize 
    if iter.γ === nothing
        if iter.L === nothing
            @warn "--> smoothness parameter absent"
            return nothing
        else
            γ = zeros(R, N)
            if isa(iter.L, R)
                γ = fill(iter.α * R(iter.N) / iter.L, (N,))
            else
                for i = 1:N
                    γ[i] = iter.α * R(N) / iter.L[i]
                end
            end
        end
    else
        isa(iter.γ, R) ? (γ = fill(iter.γ, (N,))) : (γ = iter.γ) #provided γ
    end
    # computing the gradients and updating the table s 
    # s = Vector{Tx}(undef, 0)
    s = zeros(n,N)
    av = zero(iter.x0)
    for i = 1:N
        ∇f, ~ = gradient(iter.F[i], iter.x0)
        ∇h, ~ = gradient(iter.H[i], iter.x0)
        # push!(s, ∇h / γ[i] - ∇f / N) # table of s_i
        s[:,i] = ∇h / γ[i] - ∇f / N  # table of s_i
        av += ∇h / γ[i] - ∇f / N
    end
    #initializing the vectors 
    # av = sum(s) # the running average  

    z, ~ = prox_Breg(iter.H, iter.g, av, γ)    
    state = Breg_FINITO_basic_state(s, γ, av, z, ind, d)

    println("basic verison of bregman finito")

    return state, state
end

function Base.iterate(
    iter::Breg_FINITO_basic_iterable{R},
    state::Breg_FINITO_basic_state{R},
) where {R}

    for i = 1:iter.N    # performing the main steps 
        gradient!(state.∇f_temp, iter.F[i], state.z)
        state.∇f_temp ./= iter.N
        gradient!(state.∇h_temp, iter.H[i], state.z)
        state.∇h_temp ./= state.γ[i]
        state.∇h_temp .-= state.∇f_temp
        @. state.av += state.∇h_temp - state.s[:,i]
        state.s[:,i] .= state.∇h_temp  #update s_i

        prox_Breg!(state.z, iter.H, iter.g, state.av, state.γ)
    end
    
    state.epoch_cnt += 1
    return state, state
end

solution(state::Breg_FINITO_basic_state) = state.z
solution_s(state::Breg_FINITO_basic_state) = state.s
solution_γ(state::Breg_FINITO_basic_state) = state.γ
solution_epoch(state::Breg_FINITO_basic_state) = state.epoch_cnt

## TODO list
## this is implemented in this way for speed, however, add different sweeping strategies.
## add array of H functions. Now, it is assumed H is similar for all the funcitons in prox_Breg
