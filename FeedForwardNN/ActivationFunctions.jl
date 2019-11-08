
function logistic(θ::T) where {T<:Real}
    return 1 / (1 + exp(θ))
end
