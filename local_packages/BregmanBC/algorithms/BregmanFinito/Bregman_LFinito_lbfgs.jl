# Bregman SPIRAL

struct LBreg_Finito_lbfgs_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg,TH}
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term
    H::Any                  # distance generating function 
    x0::Tx                  # initial point
    N::Int                       # number of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i    
    γ::Maybe{Union{Array{R},R}}  # stepsizes 
    β::R                    # ls division parameter for τ
    sweeping::Int8          # to only use one stepsize γ
    batch::Int              # batch size
    α::R                    # in (0, 1), e.g.: 0.99
    B::TH                   # lbfgs module
    ls_tol::R               # tolerance in ls
    D::Maybe{R}             # a large number to normalize lbfgs direction if provided
end

mutable struct LBreg_Finito_lbfgs_state{R<:Real,Tx,TH}
    γ::Array{R}             # stepsize parameter
    av::Tx                  # the running average
    ind::Array{Array{Int}}  # running index set
    d::Int                  # number of batches
    B::TH                   # lbfgs module

    # some extra placeholders  
    tz::Tx                  # \tilde z_i^k
    y::Tx                   # y^k
    ∇f_temp::Tx             # placeholder for gradients 
    temp::R                 # placeholder for bregman distances 
    z::Tx                   # z^k
    z_prev::Maybe{Tx}       # z^k previous
    v::Tx                   # v^k
    v_prev::Maybe{Tx}       # v^k previous
    dir::Tx                 # direction d^k
    ∇f_sum::Tx              # for envelope value
    u::Tx                   # linesearch candidate u^k
    inds::Array{Int}        # needed for the shuffled sweeping
    τ::Float64              # interpolation parameter between the quasi-Newton direction and the nominal step 
end

function LBreg_Finito_lbfgs_state(γ::Array{R}, av::Tx, ind, d, B::TH) where {R,Tx,TH}
    return LBreg_Finito_lbfgs_state{R,Tx,TH}(
        γ,
        av,
        ind,
        d,
        B,
        copy(av),
        copy(av),
        copy(av),
        0.0,
        copy(av),
        nothing,
        copy(av),
        nothing,
        copy(av),
        copy(av),
        copy(av),
        collect(1:d),
        1.0
    )
end

function Base.iterate(iter::LBreg_Finito_lbfgs_iterable{R}) where {R}
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
    state = LBreg_Finito_lbfgs_state(γ, av, ind, cld(N, r), iter.B)

    return state, state
end

function Base.iterate(
    iter::LBreg_Finito_lbfgs_iterable{R},
    state::LBreg_Finito_lbfgs_state{R},
) where {R}

    if state.z_prev === nothing # for lbfgs updates
        state.z_prev = zero(state.tz)
        state.v_prev = zero(state.tz)
    end

    prox_Breg!(state.z, iter.H, iter.g, state.av, state.γ)

    envVal = 0.0 # Lyapunov function value (L(v^k,z^k))
    state.∇f_sum .= zero(state.av) 
    state.av .= zero(state.z) # av = s^k

    # full update
    for i = 1:iter.N
        gradient!(state.∇f_temp, iter.H[i], state.z)
        state.av .+= state.∇f_temp ./ state.γ[i]
        state.∇f_temp, fi_z = gradient(iter.F[i], state.z) 
        state.av .-= state.∇f_temp ./ iter.N
        envVal += fi_z / iter.N
        state.∇f_sum .+= state.∇f_temp 
    end

    prox_Breg!(state.v, iter.H, iter.g, state.av, state.γ) 
    envVal += iter.g(state.v) 
    state.v .-= state.z # v - z 

    # update lbfgs
    update!(state.B, state.z - state.z_prev, -state.v +  state.v_prev) # to update B by lbfgs! (B, s-s_pre, y-y_pre)
    # store vectors for next update
    copyto!(state.z_prev, state.z)
    copyto!(state.v_prev, state.v)

    mul!(state.dir, state.B, state.v) # update d
    if iter.D != nothing # normalizing the direction if necessarily
        state.dir .= state.dir * D * norm(state.v-state.z)/norm(state.dir)
    end

    envVal += real(dot(state.∇f_sum, state.v)) / iter.N
    state.temp = dist_Breg(iter.H[1], state.v + state.z, state.z) 
    state.temp *= sum(1 ./ state.γ)
    envVal += state.temp

    state.τ = 1.0
    # linesearch
    for i=1:5 
        state.u .=  state.z .+ (1- state.τ) .* state.v + state.τ * state.dir

        state.av .= zero(state.u) # av = \tilde s^k
        state.∇f_sum .= zero(state.av)
        envVal_trial = 0  # Lyapunov function value (L(y^k,u^k))
        for i = 1:iter.N
            gradient!(state.∇f_temp, iter.H[i], state.u)
            state.av .+= state.∇f_temp ./ state.γ[i]
            state.∇f_temp, fi_z = gradient(iter.F[i], state.u) # update the gradient
            state.av .-= state.∇f_temp ./ iter.N
            envVal_trial += fi_z / iter.N
            state.∇f_sum .+= state.∇f_temp 
        end
        prox_Breg!(state.y, iter.H, iter.g, state.av, state.γ)
        envVal_trial += iter.g(state.y)
        state.y .-= state.u # y-u 

        envVal_trial += real(dot(state.∇f_sum, state.y)) / iter.N
        state.temp = dist_Breg(iter.H[1], state.y + state.u, state.u)
        state.temp *= sum(1 ./ state.γ)
        envVal_trial += state.temp

        envVal_trial <= envVal + eps(R) && break # descent on the envelope function (Table 1, 5e) # here it seems accurate precisions result in better results. No  stabiliry issues are seen.
        state.τ *= iter.β   # backtracking on τ
        println("ls on τ")
    end

    iter.sweeping == 2 && (state.inds = randperm(state.d)) # shuffled sweeping

    # inner loop
    for j in state.inds
        prox_Breg!(state.tz, iter.H, iter.g, state.av, state.γ) # av = s^k = \tilde s^k
        for i in state.ind[j] # updating the gradient
            gradient!(state.∇f_temp, iter.F[i], state.u) 
            state.av .+= (state.∇f_temp ./ iter.N)
            gradient!(state.∇f_temp, iter.F[i], state.tz) 
            state.av .-= (state.∇f_temp ./ iter.N)

            gradient!(state.∇f_temp, iter.H[i], state.u) 
            state.av .-= (state.∇f_temp ./ state.γ[i])
            gradient!(state.∇f_temp, iter.H[i], state.tz) 
            state.av .+= (state.∇f_temp ./ state.γ[i])
        end
    end

    return state, state
end

solution(state::LBreg_Finito_lbfgs_state) = state.z
epoch_count(state::LBreg_Finito_lbfgs_state) = state.τ  # number of epochs is epoch_per_iter + log_β(τ) , where τ is from ls and epoch_per_iter is 3 or 4. Refer to it_counter function in utilities_breg.jl

