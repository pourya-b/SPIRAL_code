# SPIRAL-no-ls

struct LBreg_Finito_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg}
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

mutable struct LBreg_Finito_state{R<:Real,Tx}
    γ::Array{R}             # stepsize parameter
    av::Tx                  # the running average
    ind::Array{Array{Int}}  # running index set
    d::Int                  # number of batches
    # some extra placeholders 
    z::Tx                   # \tilde z_i^k
    ∇f_temp::Tx             # placeholder for gradients 
    z_full::Tx              # z^k
    inds::Array{Int}        # needed for shuffled sweeping
end

function LBreg_Finito_state(γ::Array{R}, av::Tx, ind, d) where {R,Tx}
    return LBreg_Finito_state{R,Tx}(
        γ,
        av,
        ind,
        d,
        copy(av),
        copy(av),
        copy(av),
        collect(1:d),
    )
end

function Base.iterate(iter::LBreg_Finito_iterable{R}) where {R}
    N = iter.N
    r = iter.batch # batch size 
    # create index sets 
    ind = Vector{Vector{Int}}(undef, 0)
    d = Int(floor(N / r))
    for i = 1:d
        push!(ind, collect(r*(i-1)+1:i*r))
    end
    r * d < N && push!(ind, collect(r*d+1:N))
    # updating the stepsize 
    if iter.γ === nothing
        if iter.L === nothing
            @warn "--> smoothness parameter absent"
            return nothing
        else
            γ = zeros(R, N)
            for i = 1:N
                isa(iter.L, R) ? (γ = fill(iter.α * R(iter.N) / iter.L, (N,))) :
                (γ[i] = iter.α * R(N) / (iter.L[i]))
            end
        end
    else
        isa(iter.γ, R) ? (γ = fill(iter.γ, (N,))) : (γ = iter.γ) # provided γ
    end
    #initializing the vectors 
    av = zero(iter.x0)
    for i = 1:N
        ∇f, ~ = gradient(iter.F[i], iter.x0)
        av .-= ∇f ./ N
        ∇H, ~ = gradient(iter.H[i], iter.x0)
        av .+= ∇H ./ γ[i]
    end
    state = LBreg_Finito_state(γ, av, ind, cld(N, r))

    return state, state
end

function Base.iterate(
    iter::LBreg_Finito_iterable{R},
    state::LBreg_Finito_state{R},
) where {R}
    prox_Breg!(state.z_full, iter.H, iter.g, state.av, state.γ)

    # full update 
    state.av .= zero(state.z_full)
    for i = 1:iter.N
        gradient!(state.∇f_temp, iter.H[i], state.z_full)
        state.av .+= state.∇f_temp ./ state.γ[i]
        gradient!(state.∇f_temp, iter.F[i], state.z_full) 
        state.av .-= state.∇f_temp ./ iter.N
    end

    iter.sweeping == 3 && (state.inds = randperm(state.d)) # shuffled

    # inner loop
    for j in state.inds
        prox_Breg!(state.z, iter.H, iter.g, state.av, state.γ)

        for i in state.ind[j] # updating the gradient
            gradient!(state.∇f_temp, iter.F[i], state.z_full) 
            state.av .+= state.∇f_temp ./ iter.N
            gradient!(state.∇f_temp, iter.F[i], state.z) 
            state.av .-= state.∇f_temp ./ iter.N

            gradient!(state.∇f_temp, iter.H[i], state.z_full) 
            state.av .-= state.∇f_temp ./ state.γ[i]
            gradient!(state.∇f_temp, iter.H[i], state.z)
            state.av .+= state.∇f_temp ./ state.γ[i]
        end
    end

    return state, state
end

solution(state::LBreg_Finito_state) = state.z_full

