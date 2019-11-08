using Revise
using LinearAlgebra
using DataFrames, CSV
using Printf

pwd()
includet("./GeneralizedLinearModels/GeneralizedLinearModels.jl")
using .GeneralizedLinearModels

Xy = CSV.read("CCDataClean.csv")
colnames = names(Xy)
println(colnames)

Xy = Matrix(Xy)
Xy[:, 1] .= 1

colnames[1] = :Intercept
colnames = colnames[1:end-1]

Xy

X, y = Xy[:, 1:29], Xy[:, 30]

lp = LinearPredictor(X)
glm = GeneralizedLinearModels.GLModel(Binomial(),lp, Logit(), colnames)

fit1 = GeneralizedLinearModels.glm(X, y, Binomial(), Logit(), colnames)
GeneralizedLinearModels.FisherScoring!(fit1)

fit1

GeneralizedLinearModels.summary(fit1)

fit1.Fit.SE

fit1.Fit.𝐃

sum(fit1.Fit.𝐃)

μ = fit1.Fit.μ

L = sum(y .* log.(μ) + (1 .- y) .* log.(1 .- μ))

exp(L)

2*29 - 2*log(L)

1/(1+exp(-1.46505))





qs = range(1/n, stop=1-1/n, length=n) # no need to collect it
d = Normal() # default is mean = 0, std = 1

0.08362313385635121/2

println("Probs")
for i in 1:length(fit1.Fit.β)
    z = abs(fit1.Fit.β[i]/fit1.Fit.SE[i])
    println(2*ccdf(d, z, ))
end

2*ccdf(NormalCanon(), abs(1.96))


fit1

β = fit1.Fit.β
colnames = fit1.Model.cols
SE = fit1.Fit.SE

t1 = rpad("Variable",15," ")
t2 = rpad("β-hat", 14, " ")
t3 = rpad("SE",13," ")
t4 = rpad("Z",8," ")
t5 = rpad("P(z > |Z|)",15," ")

@printf "%s %s %s %s %s" t1 t2 t3 t4 t5
@printf("\n")

import Distributions: NormalCanon, ccdf

for i in eachindex(β, SE, colnames)
    v1 = rpad(colnames[i],12," ")
    v2 = β[i]
    v3 = SE[i]
    v4 = v2/v3
    v5 = 2*ccdf(NormalCanon(), abs(v4))
    @printf "%s % .5e % .5e % .5e % .5e " v1 v2 v3 v4 v5
    @printf("\n")
end




for i in 1:length(y)
    a = fit1.Fit.μ[i]
    b = fit1.Fit.η[i]
    println(@sprintf "%i %3.5f  %3.5f" y[i] a b)
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
