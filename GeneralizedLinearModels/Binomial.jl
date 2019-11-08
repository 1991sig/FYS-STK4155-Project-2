
mutable struct Binomial <: ExponentialFamily end

Var(::Binomial, μ::Vector{T}) where {T<:Real} = μ .* (1 .- μ)

function LogLikelihood(::Binomial, μ::Vector{T}, y::Vector{T}) where {T<:Real}
   l = sum(y .* log.(μ) + (1 .- y) .* log.(1 .- μ))
   l
end

function Likelihood(::Binomial, μ::Vector{T}, y::Vector{T}) where {T<:Real}
   l = LogLikelihood(Binomial(), μ, y)
   L = exp(l)
   L
end

function Deviance(::Binomial, μ::Vector{T}, y::Vector{T}) where {T<:Real}
   D = similar(y)
   @inbounds for i in eachindex(μ, y)
      if y[i] == 0
         D[i] = 2*log(1/ (1 - μ[i]) )
      else
         D[i] = 2*log(1/μ[i])
      end
   end
   D
end
