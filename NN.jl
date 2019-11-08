#using Revise
using RDatasets
using LinearAlgebra

#includet("./GeneralizedLinearModels/GeneralizedLinearModels.jl")
#using .GeneralizedLinearModels

iris = dataset("datasets", "iris")
species = Dict([("setosa", 0), ("versicolor", 1), ("virginica", 0)])
iris[:Species] = map(s->species[s], iris.Species)
vars = [String(i) for i in names(iris)]

iris

Xy = Matrix(iris)
Xy = hcat(ones(size(Xy)[1], 1), Xy)
Xy

X, y = Xy[:, 1:5], Xy[:, 6]
