using Revise
using RDatasets
using LinearAlgebra

includet("./GeneralizedLinearModels/GeneralizedLinearModels.jl")
using .GeneralizedLinearModels

iris = dataset("datasets", "iris")
species = Dict([("setosa", 0), ("versicolor", 1), ("virginica", 0)])
iris[:Species] = map(s->species[s], iris.Species)
vars = [String(i) for i in names(iris)]

iris

Xy = Matrix(iris)
Xy = hcat(ones(size(Xy)[1], 1), Xy)
Xy

X, y = Xy[:, 1:5], Xy[:, 6]

lp = LinearPredictor(X)
glm = GLModel(Binomial(),lp, Logit())

fit1 = GeneralizedLinearModels.glm(X, y, Binomial(), Logit())
GeneralizedLinearModels.FisherScoring!(fit1)

fit1

using Printf

for i in 1:length(y)
    a = fit1.Fit.μ[i]
    b = fit1.Fit.η[i]
    println(@sprintf "%i  %3.5f  %3.5f" y[i] a b)
end

# Residual Deviance
sum(fit1.Fit.𝐃)


LogLikelihood(Binomial(),fit1.Fit.μ,y)

μ = fit1.Fit.μ

sum(y .* log.(μ) + (1 .- y) .* log.(1 .- μ))

-145.07/2

eta1 = X[:, 1:5] * fit1.Fit.β

eta1 == fit1.Fit.η
fit1.Fit.μ == logistic(eta1)
