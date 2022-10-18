struct SMD_basic_iterable{R<:Real,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Tg}
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term
    H::Any                  # distance generating function 
    x0::Tx                  # initial point
    N::Int                  # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i    
    γ::Maybe{R}             # stepsizes 
    α::R                    # in (0, 1), e.g.: 0.99
    diminishing::Bool       # diminishing stepsize
end

mutable struct SMD_basic_state{R<:Real,Tx}
    γ::R                    # stepsize parameter
    z::Tx
    # some extra placeholders 
    s::Tx                   # temp variable
    ∇f_temp::Tx             # placeholder for gradients 
    idxr::Int               # running idx set
    epoch_cnt::R            # epoch counter 
end

function SMD_basic_state(γ::R, z::Tx, N) where {R,Tx}
    return SMD_basic_state{R,Tx}(γ, z, copy(z), copy(z), 1, 1/N)
end

function Base.iterate(iter::SMD_basic_iterable{R,C,Tx}) where {R,C,Tx}
    N = iter.N
    # updating the stepsize 
    if iter.γ === nothing 
        if iter.L === nothing
            @warn "--> smoothness parameter absent"
            return nothing
        else
            if isa(iter.L, R)
                L = iter.L
            else
                @warn "--> smoothness parameter must be a scalar..."
                L = maximum(iter.L)
            end

            if iter.diminishing
                γ = iter.α / L # bug prone!
            else
                λ = iter.α / L
                γ = iter.α * (λ / (1 + λ * L)) # bug prone!
            end
        end
    else
        isa(iter.γ, R) ? (γ = iter.γ) : (@warn "only single stepsize is supported in SMD") # provided γ
    end

    #initializing the vectors 
    idxr = rand(1:iter.N)
    ∇H, ~ = gradient(iter.H[idxr], iter.x0)
    s = ∇H ./ γ
    ∇f, ~ = gradient(iter.F[idxr], iter.x0)
    s .-= ∇f

    z, ~ = prox_Breg(iter.H, iter.g, s, γ)

    state = SMD_basic_state(γ, z, N)

    return state, state
end

function Base.iterate(iter::SMD_basic_iterable{R}, state::SMD_basic_state{R}) where {R}

    for i in 1:iter.N # bug prone!
        if iter.diminishing
            state.γ = iter.α / (iter.N * (ceil(state.epoch_cnt) * iter.L)) # bug prone!
        end
        # compute s
        gradient!(state.∇f_temp, iter.H[i], state.z) # update the gradient
        state.s .= state.∇f_temp ./ state.γ
        gradient!(state.∇f_temp, iter.F[i], state.z) # update the gradient
        state.s .-= state.∇f_temp

        prox_Breg!(state.z, iter.H, iter.g, state.s, state.γ)
        state.epoch_cnt += 1/iter.N 
    end

    return state, state
end

solution(state::SMD_basic_state) = state.z


## TODO list
## only one H is supported in SMD...
## the update rule and the stepsize selection should be double-checked
