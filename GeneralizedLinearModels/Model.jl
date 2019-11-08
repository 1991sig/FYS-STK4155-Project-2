
mutable struct GLModel{RC<:ExponentialFamily,
                       LP<:LinearPredictor,
                       g <:Link,
                       v <: Vector{Symbol}
                       } <: AbstractGLM
    d::RC # Random component from the exponential family of distributions
    lp::LP # Linear predictor Xβ
    g::g  # 𝔼(Y|X) = μ = g⁻¹(η) and η(μ) = g(μ)
    cols::v # Column names
end

function GLModel(X::Matrix{T},
                 d::ExponentialFamily,
                 l::Link,
                 c::Vector{Symbol}) where {T<:Real}
    lp = LinearPredictor(X)
    GLModel(d, lp, x, c)
end
