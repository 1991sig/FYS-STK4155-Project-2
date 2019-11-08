
mutable struct LinearPredictor{T<:Real} <: AbstractLinearPredictor
    X::Matrix{T}
    β::Vector{T}

    function LinearPredictor{T}(X::Matrix{T}, β::Vector{T}) where {T<:Real}
        nrows, ncols = size(X)
        @assert(size(β) == ncols, "No. of coefs does not match no. of vars")
        new{T}(X, β)
    end

    function LinearPredictor{T}(X::Matrix{T}) where {T<:Real}
        nrows, ncols = size(X)
        β = zeros(T, ncols)
        new{T}(X, β)
    end
end

LinearPredictor(X::Matrix, β::Vector) = LinearPredictor{Float64}(X, β)
LinearPredictor(X::Matrix{T}) where {T<:Real} = LinearPredictor{T}(X)
