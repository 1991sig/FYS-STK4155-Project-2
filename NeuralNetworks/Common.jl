

mutable struct NeuralNetwork{M<:AbstractNeuralModel,
                             P<:Parameters} <: AbstractNeuralNetwork
    Model::M
    Parameters::P
    IsFitted::Bool
end
