"""
**Poly20 kernel**
    Poly20(λ=1.0)
With a nonnegative scalar `λ`, returns the function
```math
f(x) = - λ ∑_i log(x_i)???!!!
```
"""

export Poly20


struct Poly20{R<:Real} <: BregmanFunction
    λ::R
    function Poly20{R}(λ::R) where {R<:Real}
        if any(λ .< 0)
            error("coefficients in λ must be nonnegative")
        else
            new(λ)
        end
    end
end

function (f::Poly20{R})(x::AbstractArray{R}) where {R<:Real}
    return 0.5 * norm(x)^2
end

Poly20(λ::R = 1.0) where {R<:Real} = Poly20{R}(λ)



## gradient
function gradient!(
    y::AbstractArray{T},
    h::Poly20{S},
    x::AbstractArray{T},
) where {S<:Real,R<:Real,T<:RealOrComplex{R}}
    for k in eachindex(x)
        y[k] = h.λ * x[k]
    end
    return h.λ * x
end

### proximal map 

function prox_Breg!(
    y ::AbstractArray{R},
    h::Union{Array{Poly20{R}}, Array{Poly20},Poly20},
    g::ProximalOperators.NormL1,
    x::AbstractArray{R},
    γ::Array{R},
) where {R<:Real}
    γb = 1 / sum(1 ./ γ)
    prox!(y, g, x .* γb, γb)
end


function prox_Breg!(
    y::AbstractArray{R},
    h::Union{Array{Poly20{R}}, Array{Poly20},Poly20},
    g::ProximalOperators.NormL1,
    x::AbstractArray{R},
    γ::R,
) where {R<:Real}
    prox!(y, g, x .* γ, γ)
end
 
