# SPIRAL-no-ls

struct FINITO_LFinito_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg} <: CIAO_iterable
    F::Array{Tf}            # smooth term
    g::Tg                   # nonsmooth term
    x0::Tx                  # initial point
    N::Int                  # of data points in the finite sum problem
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i
    γ::Maybe{Union{Array{R},R}}  # stepsizes
    sweeping::Int8          # sweeping strategy in the inner loop,  # 1:cyclical, 2:shuffled
    batch::Int              # batch size
    α::R                    # in (0, 1), e.g.: 0.99
end

mutable struct FINITO_LFinito_state{R<:Real,Tx}
    γ::Array{R}             # stepsize parameter
    hat_γ::R                # average γ
    av::Tx                  # the running average
    ind::Array{Array{Int}}  # running index set
    d::Int                  # number of batches
    # some extra placeholders
    z::Tx
    ∇f_temp::Tx             # placeholder for gradients
    z_full::Tx
    inds::Array{Int}        # An array containing the indices of block batches
end

function FINITO_LFinito_state(γ::Array{R}, hat_γ::R, av::Tx, ind, d) where {R,Tx}
    return FINITO_LFinito_state{R,Tx}(
        γ,
        hat_γ,
        av,
        ind,
        d,
        copy(av),
        copy(av),
        copy(av),
        collect(1:d),
    )
end

function Base.iterate(iter::FINITO_LFinito_iterable{R}) where {R} 
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
                isa(iter.L, R) ? (γ = fill(iter.α * N / iter.L, (N,))) :
                (γ[i] = iter.α * N / (iter.L[i]))
            end
        end
    else
        isa(iter.γ, R) ? (γ = fill(iter.γ, (N,))) : (γ = iter.γ) # provided γ
    end
    #initializing the vectors
    hat_γ = 1 / sum(1 ./ γ)
    av = copy(iter.x0)
    for i = 1:N
        ∇f, ~ = gradient(iter.F[i], iter.x0)
        ∇f .*= hat_γ / N
        av .-= ∇f
    end
    state = FINITO_LFinito_state(γ, hat_γ, av, ind, cld(N, r))

    return state, state
end

function Base.iterate(
    iter::FINITO_LFinito_iterable{R},
    state::FINITO_LFinito_state{R},
) where {R}
    prox!(state.z_full, iter.g, state.av, state.hat_γ)
    state.av .= state.z_full
    for i = 1:iter.N # full update
        gradient!(state.∇f_temp, iter.F[i], state.z_full) 
        state.av .-= (state.hat_γ / iter.N) .* state.∇f_temp
    end
    iter.sweeping == 2 && (state.inds = randperm(state.d)) # shuffled

    for j in state.inds       # over block index
        prox!(state.z, iter.g, state.av, state.hat_γ)
        for i in state.ind[j] # over each block of indices
            gradient!(state.∇f_temp, iter.F[i], state.z_full) # update the gradient
            state.av .+= (state.hat_γ / iter.N) .* state.∇f_temp
            gradient!(state.∇f_temp, iter.F[i], state.z) # update the gradient
            state.av .-= (state.hat_γ / iter.N) .* state.∇f_temp
            state.av .+= (state.hat_γ / state.γ[i]) .* (state.z .- state.z_full)
        end
    end

    return state, state
end

solution(state::FINITO_LFinito_state) = state.z 
