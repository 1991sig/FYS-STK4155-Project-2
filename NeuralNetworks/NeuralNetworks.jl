module NeuralNetworks

    using LinearAlgebra, Printf
    import StatsBase: sample
    import Random: MersenneTwister

    export
        # Activation Functions
        logistic,

        # Models
        FeedForwardNet,
        NeuralNetwork,
        Parameters,

        # Functions
        FeedForward!,
        FeedForward,
        Backpropagation!,
        Train!,
        Predict,
        Score

    abstract type AbstractNeuralNetwork end
    abstract type AbstractNeuralModel end


    include("ActivationFunctions.jl")
    include("NeuralNet.jl")
    include("Parameters.jl")
    include("Common.jl")
    include("Train.jl")
end
