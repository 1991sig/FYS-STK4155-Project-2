

mutable struct Parameters{T<:Real}
    𝐖ᴴ::Array{T}
    𝐛ᴴ::Array{T}
    𝐖ᴼ::Array{T}
    𝐛ᴼ::Array{T}
    𝐳ᴴ::Array{T}
    𝐚ᴴ::Array{T}
    𝐳ᴼ::Array{T}
    P::Array{T}

    function Parameters{T}(HiddenWeights::Array{T},
                           HiddenBias::Array{T},
                           OutputWeights::Array{T},
                           OutputBias::Array{T},
                           z_h::Array{T},
                           a_h::Array{T},
                           z_o::Array{T},
                           P::Array{T}) where {T <: Real}
        new{T}(HiddenWeights, HiddenBias, OutputWeights, OutputBias,
               z_h, a_h, z_o, P)
    end
end


function Parameters(random_generator::MersenneTwister,
                    n_features::Int,
                    n_hidden_neurons::Int,
                    n_categories::Int,
                    batch_size::Int) where {T <: Real}

    hw = randn(rng, Float64, (n_features, n_hidden_neurons))
    hb = zeros(Float64, n_hidden_neurons) .+ 0.01
    ow = randn(rng, Float64, (n_hidden_neurons, n_categories))
    ob = zeros(Float64, n_categories) .+ 0.01
    zh = zeros(Float64, batch_size, n_hidden_neurons)
    ah = zeros(Float64, batch_size, n_hidden_neurons)
    zo = zeros(Float64, batch_size, n_categories)
    p = zeros(Float64, batch_size, n_categories)
    Parameters{T}(hw, hb, ow, ob, zh, ah, zo, p)
end

function Parameters(random_generator::MersenneTwister,
                    NM::M) where {M<:AbstractNeuralModel}
    n_features = NM.n_features
    n_hidden_neurons = NM.n_hidden_neurons
    n_categories = NM.n_categories
    batch_size = NM.batch_size

    hw = randn(random_generator, Float64, (n_features, n_hidden_neurons))
    hb = zeros(Float64, n_hidden_neurons) .+ 0.01
    ow = randn(random_generator, Float64, (n_hidden_neurons, n_categories))
    ob = zeros(Float64, n_categories) .+ 0.01
    zh = zeros(Float64, batch_size, n_hidden_neurons)
    ah = zeros(Float64, batch_size, n_hidden_neurons)
    zo = zeros(Float64, batch_size, n_categories)
    p = zeros(Float64, batch_size, n_categories)
    Parameters{Float64}(hw, hb, ow, ob, zh, ah, zo, p)
end
