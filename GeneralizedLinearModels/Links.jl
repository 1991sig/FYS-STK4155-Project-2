
mutable struct Logit <: Link end

function LinkFunction end
function InverseLinkFunction end
function CanonicalLink end

LinkFunction(::Logit, μ::Vector{T}) where {T<:Real} = logit(μ)
InverseLinkFunction(::Logit, η::Vector{T}) where {T<:Real} = logistic(η)
CanonicalLink(::Binomial) = Logit()
