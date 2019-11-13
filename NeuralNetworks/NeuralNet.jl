
mutable struct FeedForwardNet{T<:Real} <: AbstractNeuralModel
    X_data::Matrix{T}
    Y_data::VecOrMat{T}
    n_hidden_neurons::Int
    n_categories::Int
    epochs::Int
    batch_size::Int
    η::Float64
    λ::Float64
    n_inputs::Int
    n_features::Int
    iterations::Int

    function FeedForwardNet{T}(X_data::Matrix{T},
                               Y_data::VecOrMat{T},
                               n_hidden_neurons::Int=50,
                               n_categories::Int=1,
                               epochs::Int=10,
                               batch_size::Int=100,
                               η::Float64=0.1,
                               λ::Float64=0.0) where {T <: Real}

        n_inputs, n_features = size(X_data)
        iterations = n_inputs ÷ batch_size

        new{T}(X_data,
               Y_data,
               n_hidden_neurons,
               n_categories,
               epochs,
               batch_size,
               η,
               λ,
               n_inputs,
               n_features,
               iterations)
    end
end
