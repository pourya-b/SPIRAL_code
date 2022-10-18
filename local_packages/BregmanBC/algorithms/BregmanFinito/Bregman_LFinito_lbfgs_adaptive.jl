# Bregman adaSPIRAL 

struct LBreg_Finito_lbfgs_adaptive_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg,TH}
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term
    H::Any                  # distance generating function 
    x0::Tx                  # initial point
    N::Int                       # number of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i    
    γ::Maybe{Union{Array{R},R}}  # stepsizes 
    η::R                    # ls parameter for γ
    β::R                    # ls parameter for τ
    sweeping::Int8          # to only use one stepsize γ
    batch::Int              # batch size
    α::R                    # in (0, 1), e.g.: 0.99
    B::TH                   # lbfgs module
    ls_tol::R               # tolerance in ls
    D::Maybe{R}             # a large number to normalize lbfgs direction if provided
end

mutable struct LBreg_Finito_lbfgs_adaptive_state{R<:Real,Tx,TH}
    γ::Array{R}             # stepsize parameter
    s::Tx                   # the running average (vector s)
    ts::Tx                  # the running average (vector \tilde s)
    bs::Tx                  # the running average (vector \bar s)
    ind::Array{Array{Int}}  # running index set
    d::Int                  # number of batches
    B::TH                   # lbfgs module
    inds::Array{Int}        # needed for the shuffled sweeping
    τ::Float64              # interpolation parameter between the quasi-Newton direction and the nominal step

    # iterators
    z::Tx                   # z^k
    v::Tx                   # v^k
    u::Tx                   # linesearch candidate u^k
    y::Tx                   # y^k
    tz::Tx                  # \tilde z_i^k
    dir::Tx                 # direction d^k

    # vectors
    sum_∇fz::Tx
    sum_∇Hz::Tx
    sum_∇fu::Tx
    sum_∇Hu::Tx
    ∇fu::Tx
    ∇Hu::Tx
    ∇f_temp::Tx             # placeholder for gradients 
    temp::Tx                # placeholder for bregman distances 
    z_prev::Maybe{Tx}       # z^k previous
    v_prev::Maybe{Tx}       # v^k previous
        
    # scalars
    sum_fz::R
    sum_hz::R
    sum_hv::R
    sum_fu::R
    sum_hu::R
    sum_hy::R
    sum_ftz::R
    sum_htz::R
    sum_innprod_∇Htz::R
    sum_innprod_∇ftz::R
    sum_innprod_∇Hz::R
    sum_innprod_∇fz::R
    sum_innprod_∇Hu::R
    sum_innprod_∇fu::R
    innprod_∇Hu::R
    innprod_∇fu::R

    # function velues
    hu::R
    fu::R
    ftz::R
    htz::R
    val_gv::R
    val_gy::R

    first_flag::Bool
    ls_grad_eval::Int       # number of grad evaluations in the backtrackings of the stepsize (in ls 3)

    test::R
end

function LBreg_Finito_lbfgs_adaptive_state(γ, s0::Tx, ind, d, B::TH, sum_ftz::R, sum_htz::R, sum_innprod_∇ftz::R, sum_innprod_∇Htz::R, sum_∇fu, sum_∇Hu) where {R,Tx,TH}
    return LBreg_Finito_lbfgs_adaptive_state{R,Tx,TH}(
        γ,
        s0,
        s0,
        s0,
        ind,
        d,
        B,
        collect(1:d),
        1.0,
        # iterators
        zero(s0),
        zero(s0),
        zero(s0),
        zero(s0),
        zero(s0),
        zero(s0),
        # vectors
        zero(s0),
        zero(s0),
        sum_∇fu,
        sum_∇Hu,
        zero(s0),
        zero(s0),
        zero(s0),
        zero(s0),
        nothing,
        nothing,
        # scalars
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        sum_ftz,
        sum_htz,
        sum_innprod_∇Htz,
        sum_innprod_∇ftz,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        # function velues
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,

        true,
        0,

        0.0,
    )
end

function Base.iterate(iter::LBreg_Finito_lbfgs_adaptive_iterable{R}) where {R}
    N = iter.N
    r = iter.batch # batch size 
    # create index sets 
    ind = Vector{Vector{Int}}(undef, 0)
    d = Int(floor(N / r))
    for i = 1:d
        push!(ind, collect(r*(i-1)+1:i*r))
    end
    r * d < N && push!(ind, collect(r*d+1:N))

    #initializing the vectors
    sum_ftz = R(0)
    sum_innprod_∇ftz = R(0)
    sum_∇fu = zero(iter.x0)
    sum_∇Hu_no_γ = zero(iter.x0)

    for i = 1:N
        sum_ftz += iter.F[i](iter.x0)

        ∇f, ~ = gradient(iter.F[i], iter.x0)
        sum_∇fu .+= ∇f
        ∇H, ~ = gradient(iter.H[i], iter.x0)
        sum_∇Hu_no_γ .+= ∇H 
    end
    sum_innprod_∇ftz = real(dot(sum_∇fu, iter.x0))


    # updating the stepsize 
    if iter.γ === nothing
        if iter.L === nothing
            xeps = iter.x0 .+ one(R)
            av_eps = zero(iter.x0)
            avh_eps = zero(iter.x0)

            for i = 1:N
                ∇f, ~ = gradient(iter.F[i], xeps)
                av_eps .+= ∇f
                ∇H, ~ = gradient(iter.H[i], xeps)
                avh_eps .+= ∇H 
            end
            nmg = norm(sum_∇fu - av_eps)
            dmg = norm(sum_∇Hu_no_γ - avh_eps)
            t = 1
            while nmg < eps(R) || dmg < eps(R) 
                println("initial upper bound for L is numerically unstable")
                xeps = iter.x0 .+ rand(t * [-1, 1], size(iter.x0))
                av_eps = zero(iter.x0)
                avh_eps = zero(iter.x0)
    
                for i = 1:N
                    ∇f, ~ = gradient(iter.F[i], xeps)
                    av_eps .+= ∇f
                    ∇H, ~ = gradient(iter.H[i], xeps)
                    avh_eps .+= ∇H 
                end
                nmg = norm(sum_∇fu - av_eps)
                dmg = norm(sum_∇Hu_no_γ - avh_eps)
                t *= 2
            end
            L_int = nmg / dmg
            γ = iter.N * iter.α / (L_int)
            γ = fill(γ, (N,)) # to make it a vector
            println("γ specified by L_int")
        else
            γ = 10 * iter.α * R(iter.N) / (minimum(iter.L))
            γ = fill(γ, (N,)) # to make it a vector
            println("gamma specified by min{L}/10") # This choice of the stepsize seems to give more stable results over the tested datasets and problems. Due to stability issues, the algorithm is sensetive against stepsize initialization
        end
    else
        γ_max = maximum(iter.γ)
        γ = fill(γ_max, (N,)) # provided γ
        println("gamma specified by max{γ}")
    end

    #initializing the vectors

    sum_htz = R(0)
    sum_innprod_∇Htz = R(0)
    sum_∇Hu = zero(iter.x0)
    for i = 1:N
        sum_htz += iter.H[i](iter.x0)/γ[i]
        ∇H, ~ = gradient(iter.H[i], iter.x0)
        sum_∇Hu .+= ∇H ./ γ[i]
    end
    sum_innprod_∇Htz = real(dot(sum_∇Hu, iter.x0))
    s0 = sum_∇Hu - sum_∇fu/N

    state = LBreg_Finito_lbfgs_adaptive_state(γ, s0, ind, cld(N, r), iter.B, sum_ftz, sum_htz, sum_innprod_∇ftz, sum_innprod_∇Htz, sum_∇fu, sum_∇Hu) # s0 is bug
    return state, state
end

function Base.iterate(
    iter::LBreg_Finito_lbfgs_adaptive_iterable{R},
    state::LBreg_Finito_lbfgs_adaptive_state{R},
) where {R}

    if state.z_prev === nothing # for lbfgs updates
        state.z_prev = zero(iter.x0)
        state.v_prev = zero(iter.x0)
    end

    while true # first ls
        prox_Breg!(state.z, iter.H, iter.g, state.s, state.γ) # z^k in Alg. 1&2

        state.sum_fz = 0 # for ls 1
        state.sum_hz = 0
        for i in 1:iter.N 
            state.sum_fz += iter.F[i](state.z)
            state.sum_hz += iter.H[i](state.z)/state.γ[i]
        end 

        # lhs and rhs of the ls condition (Table 1, 1b)
        ls_lhs = state.sum_fz #*
        ls_lhs -= real(dot(state.sum_∇fu, state.z)) # sum_∇fu = \sum ∇f_i(\tilde z_i^k)
        ls_lhs -= state.sum_ftz 
        ls_lhs += state.sum_innprod_∇ftz

        temp = state.sum_hz + state.sum_innprod_∇Htz
        ls_rhs = (1 + iter.ls_tol) * temp
        # this choice of tolerance seems more stable, this ls is different from the others as we have already some parameters computed in the previous iteration. For instance, recalculating state.sum_htz is different from updating it with state.sum_htz ./= iter.η. Numerical instability occurs more when we have divisions with γ[i], when these stepsizes are very small due to the backtracks during the linesearchs.
        ls_rhs -= real(dot(state.sum_∇Hu, state.z)) 
        ls_rhs -= state.sum_htz 
        ls_rhs *= iter.N * iter.α

        ls_lhs <= ls_rhs + iter.ls_tol  && break  # the ls condition (Table 1, 1b)
        println("ls 1")

        state.γ *= iter.η
        # update s^k
        state.s .= state.s/iter.η + (1/iter.η - 1) * state.sum_∇fu/iter.N
        state.sum_htz /= iter.η
        state.sum_∇Hu ./= iter.η
        state.sum_innprod_∇Htz /= iter.η

        # lbfgs reset
        reset!(state.B) 
        state.z_prev .= zero(state.s) # we have to update these as well, as it is not considered in the reset function. This update causes instability though!
        state.v_prev .= zero(state.s)
    end 
    # now z^k is fixed, moving to step 2
    state.bs .= zero(state.z) # bs = \bar s^k

    # full update
    state.sum_∇fz .= zero(state.s) # for ls 2 and \bar s^k
    state.sum_∇Hz .= zero(state.s) # for ls 2 and \bar s^k
    for i = 1:iter.N
        gradient!(state.∇f_temp, iter.H[i], state.z)
        state.bs .+= state.∇f_temp ./ state.γ[i]
        state.sum_∇Hz .+= state.∇f_temp ./ state.γ[i]
        gradient!(state.∇f_temp, iter.F[i], state.z) 
        state.bs .-= state.∇f_temp ./ iter.N
        state.sum_∇fz .+= state.∇f_temp
    end
    state.sum_innprod_∇fz = real(dot(state.sum_∇fz, state.z)) # for ls 2
    state.sum_innprod_∇Hz = real(dot(state.sum_∇Hz, state.z)) # for ls 2

    while true # second ls
        prox_Breg!(state.v, iter.H, iter.g, state.bs, state.γ)  # v^k and g(v)
        state.val_gv = iter.g(state.v)

        sum_fv = 0 # for ls 2
        state.sum_hv = 0 # for ls 2
        for i in 1:iter.N 
            sum_fv += iter.F[i](state.v)
            state.sum_hv += iter.H[i](state.v)/state.γ[i]
        end

        ls_lhs = sum_fv
        ls_lhs -= state.sum_fz 
        ls_lhs -= real(dot(state.sum_∇fz,state.v))
        ls_lhs += state.sum_innprod_∇fz
        
        ls_rhs = state.sum_hv 
        ls_rhs -= state.sum_hz 
        ls_rhs -= real(dot(state.sum_∇Hz,state.v))
        ls_rhs += state.sum_innprod_∇Hz
        ls_rhs *= iter.N

        ls_lhs <= ls_rhs + iter.ls_tol && break  # the ls condition (Table 1, 3b) # the tolerance should not be eps(R) necessarily as the arithmetic operations in lhs and rhs of the condition causes tolerances more than machine precision.
        println("ls 2")

        state.γ *= iter.η
        # update bs^k
        state.bs .= state.bs/iter.η + (1/iter.η - 1) * state.sum_∇fz/iter.N
        state.sum_hz /= iter.η
        state.sum_∇Hz ./= iter.η
        state.sum_innprod_∇Hz /= iter.η

        reset!(state.B) 
        state.z_prev .= zero(state.s)
        state.v_prev .= zero(state.s)
    end   
    # now v^k and bs are fixed, moving to step 4

    # update lbfgs
    update!(state.B, state.z - state.z_prev, state.z-state.v +  state.v_prev) 
    # store vectors for next update
    copyto!(state.z_prev, state.z)
    copyto!(state.v_prev, state.v-state.z)
    mul!(state.dir, state.B, state.v-state.z) # updating the quasi-Newton direction
    
    # resetting H with zeroing previous vectors in the ls sometimes results in instability and generating large dir. To avoid very large dir and then linesearch on τ, we normalize the dir vector.
    if iter.D != nothing # normalizing the direction if necessarily
        state.dir .= state.dir * D * norm(state.v-state.z)/norm(state.dir)
    end

    # prepare for linesearch on τ
    envVal = state.sum_fz / iter.N #* # envelope value (lyapunov function) L(v^k,z^k)
    envVal += state.val_gv #*
    envVal -= state.sum_hz
    envVal += state.sum_hv
    envVal -= state.sum_innprod_∇fz / iter.N #*
    envVal += real(dot(state.sum_∇fz/iter.N, state.v)) #*
    envVal -= real(dot(state.sum_∇Hz, state.v)) 
    envVal += state.sum_innprod_∇Hz
   
    state.τ = 1
    for i=1:5 # backtracking on τ
        while true # third ls
            state.u .=  state.z .+ (1- state.τ) .* (state.v .- state.z) + state.τ * state.dir # u^k

            state.sum_fu = 0 # for ls 3
            state.sum_hu = 0 # for ls 3
            for i in 1:iter.N 
                state.sum_fu += iter.F[i](state.u)
                state.sum_hu += iter.H[i](state.u)/state.γ[i]
            end

            state.ts .= zero(state.u) # ts = \tlde s^k
            # full update
            state.sum_∇fu .= zero(state.ts) # for ls 3 and \bar s^k
            state.sum_∇Hu .= zero(state.ts) # for ls 3 and \bar s^k
            for i = 1:iter.N
                gradient!(state.∇f_temp, iter.H[i], state.u)
                state.ts .+= state.∇f_temp ./ state.γ[i]
                state.sum_∇Hu .+= state.∇f_temp ./ state.γ[i]
                gradient!(state.∇f_temp, iter.F[i], state.u) 
                state.ts .-= state.∇f_temp ./ iter.N
                state.sum_∇fu .+= state.∇f_temp
            end
            state.sum_innprod_∇fu = real(dot(state.sum_∇fu, state.u)) # for ls 3
            state.sum_innprod_∇Hu = real(dot(state.sum_∇Hu, state.u)) # for ls 3
            
            prox_Breg!(state.y, iter.H, iter.g, state.ts, state.γ) # y^k and g(y)
            state.val_gy = iter.g(state.y)
    
            sum_fy = 0 # for ls 3
            state.sum_hy = 0 # for ls 3
            for i in 1:iter.N 
                sum_fy += iter.F[i](state.y)
                state.sum_hy += iter.H[i](state.y)/state.γ[i]
            end
    
            ls_lhs = sum_fy
            ls_lhs -= state.sum_fu
            ls_lhs -= real(dot(state.sum_∇fu,state.y))
            ls_lhs += state.sum_innprod_∇fu
            
            ls_rhs = state.sum_hy
            ls_rhs -= state.sum_hu
            ls_rhs -= real(dot(state.sum_∇Hu,state.y))
            ls_rhs += state.sum_innprod_∇Hu
            ls_rhs *= iter.N
    
            ls_lhs <= ls_rhs + iter.ls_tol && break  # the ls condition (Table 1, 5d) # the tolerance should not be eps(R) necessarily as the arithmetic operations in lhs and rhs of the condition causes tolerances more than machine precision.
            println("ls 3")

            state.ls_grad_eval += 1 # here we incur extra gradient evaluations, to be counted
            
            state.γ *= iter.η
            state.bs .= state.bs/iter.η + (1/iter.η - 1) * state.sum_∇fz/iter.N  # update bs^k
            state.sum_hz /= iter.η
            state.sum_∇Hz ./= iter.η
            state.sum_innprod_∇Hz /= iter.η

            prox_Breg!(state.v, iter.H, iter.g, state.bs, state.γ) # updating v
            state.val_gv = iter.g(state.v)
            state.sum_hv = 0 # updating
            for i in 1:iter.N 
                state.sum_hv += iter.H[i](state.v)/state.γ[i]
            end

            # resetting τ
            state.τ = 1

            # reseting lbfgs
            reset!(state.B) # bug prone!
            state.z_prev .= zero(state.s)
            state.v_prev .= zero(state.s)

            # update lbfgs and the direction (bug prone!)
            update!(state.B, state.z - state.z_prev, state.z-state.v +  state.v_prev) 
            copyto!(state.z_prev, state.z)
            copyto!(state.v_prev, state.v-state.z)
            mul!(state.dir, state.B, state.v-state.z)  
            if iter.D != nothing
                state.dir .= state.dir * D * norm(state.v-state.z)/norm(state.dir)
            end     

            # updating the lyapunov function L(v^k,z^k)
            envVal = state.sum_fz / iter.N  
            envVal += state.val_gv 
            envVal -= state.sum_hz
            envVal += state.sum_hv
            envVal -= state.sum_innprod_∇fz / iter.N 
            envVal += real(dot(state.sum_∇fz/iter.N, state.v)) 
            envVal -= real(dot(state.sum_∇Hz, state.v)) 
            envVal += state.sum_innprod_∇Hz 
        end

        envVal_trial = state.sum_fu / iter.N  
        envVal_trial += state.val_gy 
        envVal_trial -= state.sum_hu
        envVal_trial += state.sum_hy
        envVal_trial -= state.sum_innprod_∇fu / iter.N
        envVal_trial += real(dot(state.sum_∇fu/iter.N, state.y)) 
        envVal_trial -= real(dot(state.sum_∇Hu, state.y)) 
        envVal_trial += state.sum_innprod_∇Hu
        
        envVal_trial <= envVal + eps(R) && break # descent on the envelope function (Table 1, 5e) # here it seems accurate precisions result in better results. No  stabiliry issues are seen.
        state.τ *= iter.β   # backtracking on τ
        println("ls on τ")
    end
    state.s .= state.ts

    iter.sweeping == 2 && (state.inds = randperm(state.d)) # shuffled sweeping

    # resetting necessary parameters for ls 1
    state.sum_ftz = R(0)
    state.sum_innprod_∇ftz = R(0)
    state.sum_htz = R(0)
    state.sum_innprod_∇Htz = R(0)

    # inner loop
    for j in state.inds # batch indices
        for i in state.ind[j] # in Algorithm 1 line 7, batchsize is 1, but here we are more general - state.ind: indices in the j th batch
            state.fu = iter.F[i](state.u)
            gradient!(state.∇fu, iter.F[i], state.u) 
            state.hu = iter.H[i](state.u)/state.γ[i]
            gradient!(state.∇Hu, iter.H[i], state.u) 
            state.∇Hu ./= state.γ[i]
            state.innprod_∇fu = real(dot(state.∇fu, state.u))
            state.innprod_∇Hu = real(dot(state.∇Hu, state.u))

            while true # forth ls
                prox_Breg!(state.tz, iter.H, iter.g, state.s, state.γ)  # tz^k and g(tz)
        
                state.ftz = iter.F[i](state.tz)
                state.htz = iter.H[i](state.tz)/state.γ[i]

                ls_lhs = state.ftz
                ls_lhs -= state.fu
                ls_lhs -= real(dot(state.∇fu, state.tz))
                ls_lhs += state.innprod_∇fu
                
                ls_rhs = state.htz 
                ls_rhs -= state.hu
                ls_rhs -= real(dot(state.∇Hu, state.tz))
                ls_rhs += state.innprod_∇Hu
                ls_rhs *= iter.N
        
                ls_lhs <= ls_rhs + iter.ls_tol && break  # the ls condition (Table 1, 8b)                
                println("ls 4")
        
                γ_prev = state.γ[i]
                state.γ[i] *= iter.η
                # update s^k
                ∇Hu = zero(state.u)
                gradient!(∇Hu, iter.H[i], state.u)
                state.s .+= (1/state.γ[i] - 1/γ_prev) * ∇Hu

                state.hu /= iter.η
                state.∇Hu ./= iter.η
                state.innprod_∇Hu /= iter.η
        
                reset!(state.B) # bug prone!
                state.z_prev .= zero(state.s)
                state.v_prev .= zero(state.s)
            end  

            # updates for f 
            state.s .+= (state.∇fu ./ iter.N)
            state.sum_∇fu .-= state.∇fu
            gradient!(state.∇f_temp, iter.F[i], state.tz) 
            state.s .-= (state.∇f_temp ./ iter.N)
            state.sum_∇fu .+= state.∇f_temp
            # updates for H
            state.s .-= state.∇Hu
            state.sum_∇Hu .-= state.∇Hu
            gradient!(state.temp, iter.H[i], state.tz) 
            state.temp ./= state.γ[i]
            state.s .+= state.temp
            state.sum_∇Hu .+= state.temp

            # updates for the next linesearch (ls 1). Sometimes these parameters are very large (due to small γ), resulting in many instability issues.
            state.sum_ftz += state.ftz
            state.sum_innprod_∇ftz += real(dot(state.∇f_temp, state.tz))
            state.sum_htz += state.htz # these causes instability in ls 1 especially when γ is very small
            state.sum_innprod_∇Htz += real(dot(state.temp, state.tz)) # these causes instability in ls 1 especially when γ is very small
        end
    end

    return state, state
end

solution(state::LBreg_Finito_lbfgs_adaptive_state) = state.z
epoch_count(state::LBreg_Finito_lbfgs_adaptive_state) = state.τ  # number of epochs is epoch_per_iter + log_β(τ) , where τ is from ls and epoch_per_iter is 3 or 4. Refer to it_counter function in utilities_breg.jl

