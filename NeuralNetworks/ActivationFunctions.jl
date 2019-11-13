
function logistic end

function logistic(θ::Vector{T}) where {T<:Real}
    return 1 ./ (1 .+ exp.(-θ))
end

function logistic(θ::Matrix{T}) where {T<:Real}
    l = zeros(Float64, size(θ))
    @inbounds for i in eachindex(θ)
        l[i] = 1 / (1 + exp(-θ[i]))
    end
    l
end
