mutable struct Fit{T<:Real}
    y::Vector{T}    # Response vector
    β::Vector{T}    # β-hat, the estimated coefficients
    𝐃::Vector{T}   # Residual deviance
    SE::Vector{T}   # Standard errors of β-hats
    η::Vector{T}    # η-hat, = X*β-hat
    μ::Vector{T}    # μ-hat, fitted μ
    DoF::T          # Degrees of freedom
    AIC::T          # Akaike Information Criterion
end

function Fit(y::Vector{T}, β::Vector{T}) where {T<:Real}
    nrow = size(y)
    ncol = size(β)
    D = similar(y, nrow)
    se = similar(y, ncol)
    eta = similar(y, nrow)
    mu = similar(y, nrow)
    dof = 0.0
    aic = 0.0
    return Fit{T}(y, β, D, se, eta, mu, dof, aic)
end

mutable struct GLMFit{M<:GLModel, F<:Fit}
    Model::M
    Fit::F
    IsFitted::Bool
end

function glm(X::Matrix{T},
             y::Vector{T},
             d::ExponentialFamily,
             l::Link,
             c::Vector{Symbol}) where {T<:Real}
    lp  = LinearPredictor(X)
    glm = GLModel(d, lp, l, c)
    fit = Fit(y, lp.β)
    glmfit = GLMFit(glm, fit, false)
    return glmfit
end

function glm(X::Matrix{T},
             y::Vector{T},
             d::ExponentialFamily,
             l::Link) where {T<:Real}
    p = size(X)[2]
    c = p > 1 ? vcat(Symbol("Intercept"), 
                     [Symbol("X", i) for i in 2:p]) : [Symbol("Intercept")]
    lp  = LinearPredictor(X)
    glm = GLModel(d, lp, l, c)
    fit = Fit(y, lp.β)
    glmfit = GLMFit(glm, fit, false)
    return glmfit
end

function Predict(M::GLMFit, X)
    η = X * M.Fit.β
    μ = InverseLinkFunction(M.Model.g, η)
    return μ
end
