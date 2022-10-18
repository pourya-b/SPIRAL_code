# Finito/MISO
# Block-coordinate and incremental aggregated proximal
# gradient methods for nonsmooth nonconvex problems

struct FINITO_basic_iterable{R<:Real,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Tg} <: CIAO_iterable
    F::Array{Tf}            # smooth term
    g::Tg                   # nonsmooth term
    x0::Tx                  # initial point
    N::Int                  # number of data points in the finite sum problem
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i
    γ::Maybe{Union{Array{R},R}}  # stepsizes
    sweeping::Int8          # sweeping strategy in the inner loop,  # 1:cyclical, 2:shuffled
    batch::Int              # batch size
    α::R                    # in (0, 1), e.g.: 0.99
end

mutable struct FINITO_basic_state{R<:Real,Tx}
    s::Array{Tx}            # table of x_j- γ_j/N nabla f_j(x_j)
    γ::Array{R}             # stepsize parameters
    hat_γ::R                # average γ
    av::Tx                  # the running average
    z::Tx
    ind::Array{Array{Int}} # running index set
    # some extra placeholders
    d::Int                  # number of batches
    ∇f_temp::Tx             # placeholder for gradients
    idxr::Int               # running idx in the iterate
    idx::Int                # location of idxr in 1:N
    inds::Array{Int}        # needed for shuffled only
end

function FINITO_basic_state(s, γ, hat_γ::R, av::Tx, z::Tx, ind, d) where {R,Tx} #? why it is not inside?
    return FINITO_basic_state{R,Tx}(
        s,
        γ,
        hat_γ,
        av,
        z,
        ind,
        d,
        zero(av),
        Int(1),
        Int(0),
        collect(1:d),
    )
end

function Base.iterate(iter::FINITO_basic_iterable{R,C,Tx}) where {R,C,Tx}
    N = iter.N
    # define the batches
    r = iter.batch # batch size
    # create index sets
    if iter.sweeping == 1
        ind = [collect(1:r)] # placeholder
    else
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
            for i = 1:N
                isa(iter.L, R) ? (γ = fill(iter.α * R(iter.N) / iter.L, (N,))) :
                (γ[i] = iter.α * R(N) / iter.L[i])
            end
        end
    else
        isa(iter.γ, R) ? (γ = fill(iter.γ, (N,))) : (γ = iter.γ) #provided γ
    end
    # computing the gradients and updating the table s
    s = Vector{Tx}(undef, 0)
    for i = 1:N
        ∇f, ~ = gradient(iter.F[i], iter.x0)
        push!(s, iter.x0 - γ[i] / N * ∇f) # table of x_i
    end
    #initializing the vectors
    hat_γ = 1 / sum(1 ./ γ)
    av = hat_γ * (sum(s ./ γ)) # the running average
    z, ~ = prox(iter.g, av, hat_γ)

    state = FINITO_basic_state(s, γ, hat_γ, av, z, ind, d)

    return state, state
end

function Base.iterate( 
    iter::FINITO_basic_iterable{R},
    state::FINITO_basic_state{R},
) where {R}

    for i = 1:iter.N
        gradient!(state.∇f_temp, iter.F[i], state.z)
        state.∇f_temp .*= - Float64(state.γ[i] / iter.N)
        state.∇f_temp .+= state.z
        @. state.av += (state.∇f_temp - state.s[i]) * Float64(state.hat_γ / state.γ[i])
        state.s[i] .= state.∇f_temp  #update x_i

        prox!(state.z, iter.g, state.av, state.hat_γ)
    end
    return state, state
end

solution(state::FINITO_basic_state) = state.z

## TODO list
## this is implemented in this way for speed, however, add different sweeping strategies.
