
function AIC(k::Int, l::Float64)
    return 2*k - 2*l
end

function logistic(θ::Vector{T}) where {T<:Real}
    return 1 ./ (1 .+ exp.(-θ))
end

function logit(θ::Vector{T}) where {T<:Real}
    return log.(θ ./ (1 .- θ))
end

Score(y, y_hat) = sum(y .== y_hat)/size(y)[1]
