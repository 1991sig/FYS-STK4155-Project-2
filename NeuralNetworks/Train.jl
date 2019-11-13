

function FeedForward!(X, 𝐖ᴴ, 𝐛ᴴ, 𝐖ᴼ, 𝐛ᴼ, 𝐳ᴴ, 𝐚ᴴ, 𝐳ᴼ, P)
    # Feed-forward for training
    𝐳ᴴ .= X * 𝐖ᴴ .+ 𝐛ᴴ'
    𝐚ᴴ .= logistic(𝐳ᴴ)
    𝐳ᴼ .= 𝐚ᴴ * 𝐖ᴼ .+ 𝐛ᴼ'

    exp_term = exp.(𝐳ᴼ)
    P .= size(𝐛ᴼ)[1] == 1 ? exp_term : exp_term ./ sum(exp_term, dims=2)
end

function FeedForward(X, 𝐖ᴴ, 𝐛ᴴ, 𝐖ᴼ, 𝐛ᴼ)
    # Feed-forward for output
    𝐳ᴴ = X * 𝐖ᴴ .+ 𝐛ᴴ'
    𝐚ᴴ = logistic(𝐳ᴴ)
    𝐳ᴼ = 𝐚ᴴ * 𝐖ᴼ .+ 𝐛ᴼ'

    exp_term = exp.(𝐳ᴼ)
    P = size(𝐛ᴼ)[1] == 1 ? exp_term : exp_term ./ sum(exp_term, dims=2)
    P
end


function Backpropagation!(X, y, 𝐖ᴴ, 𝐛ᴴ, 𝐖ᴼ, 𝐛ᴼ, 𝐚ᴴ, P, λ, η)
    δᴼ = P - y
    Δᴴ = δᴼ * 𝐖ᴼ' .* 𝐚ᴴ .* (1 .- 𝐚ᴴ)

    ∇𝐖ᴼ = 𝐚ᴴ' * δᴼ
    ∇𝐛ᴼ = sum(δᴼ, dims=1)

    ∇𝐖ᴴ = X' * Δᴴ
    ∇𝐛ᴴ = sum(Δᴴ, dims=1)

    if λ > 0.0
        ∇𝐖ᴼ .= ∇𝐖ᴼ .+ λ .* 𝐖ᴼ
        ∇𝐖ᴴ .= ∇𝐖ᴴ .+ λ .* 𝐖ᴴ
    end

    𝐖ᴼ = 𝐖ᴼ .- η .* ∇𝐖ᴼ
    𝐛ᴼ = 𝐛ᴼ .- η .* ∇𝐛ᴼ

    𝐖ᴴ = 𝐖ᴴ .- η .* ∇𝐖ᴴ
    𝐛ᴴ = 𝐛ᴴ .- η .* ∇𝐛ᴴ
end


function Train!(NN::NeuralNetwork, rng::MersenneTwister)
    if NN.IsFitted
        println("The model has already been fitted")
        return NN
    end

    inds = 1:NN.Model.n_inputs

    𝐖ᴴ = NN.Parameters.𝐖ᴴ
    𝐛ᴴ = NN.Parameters.𝐛ᴴ
    𝐖ᴼ = NN.Parameters.𝐖ᴼ
    𝐛ᴼ = NN.Parameters.𝐛ᴼ
    𝐳ᴴ = NN.Parameters.𝐳ᴴ
    𝐚ᴴ = NN.Parameters.𝐚ᴴ
    𝐳ᴼ = NN.Parameters.𝐳ᴼ
    P = NN.Parameters.P
    λ = NN.Model.λ
    η = NN.Model.η

    for i in 1:NN.Model.epochs
        for j in 1:NN.Model.iterations
            # Sample rownumbers for the batch
            S = sample(rng, inds, NN.Model.batch_size, replace=false)

            # Get rows from trainingdata
            X = NN.Model.X_data[S, : ]
            y = NN.Model.Y_data[S, : ]

            FeedForward!(X, 𝐖ᴴ, 𝐛ᴴ, 𝐖ᴼ, 𝐛ᴼ, 𝐳ᴴ, 𝐚ᴴ, 𝐳ᴼ, P)
            Backpropagation!(X, y, 𝐖ᴴ, 𝐛ᴴ, 𝐖ᴼ, 𝐛ᴼ, 𝐚ᴴ, P, λ, η)
        end
    end
    NN.IsFitted = true
    NN
end


function Predict(NN::NeuralNetwork, X::Matrix{T}) where {T<:Real}
    if !NN.IsFitted
        throw(ErrorException("The model has not been fitted. Fit it first."))
    end

    ps = NN.Parameters
    𝐖ᴴ, 𝐛ᴴ, 𝐖ᴼ, 𝐛ᴼ = ps.𝐖ᴴ, ps.𝐛ᴴ, ps.𝐖ᴼ, ps.𝐛ᴼ

    P = FeedForward(X, 𝐖ᴴ, 𝐛ᴴ, 𝐖ᴼ, 𝐛ᴼ)
    if NN.Model.n_categories == 1
        return map(p -> ifelse(p >= 0.5, 1, 0), P)
    end

    return map(s -> s[2], argmax(P, dims=2))
end

Score(y, y_hat) = sum(y .== y_hat)/size(y)[1]
