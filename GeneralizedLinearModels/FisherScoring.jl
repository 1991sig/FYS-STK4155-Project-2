

function Score end
Score(X, y, μ, ::Binomial, ::Logit) = X' * (y - μ)

function FisherInformation end
FisherInformation(X, w, ::Binomial, ::Logit) = X' * diagm(w) * X

function FisherScoring!(M::GLMFit, maxiter=25, tol=1e-08)
        dist, lp, link = M.Model.d, M.Model.lp, M.Model.g
        X = lp.X
        β = M.Fit.β
        μ = M.Fit.μ
        y = M.Fit.y
        η = M.Fit.η
        p = length(β)

        local J, Jinv, w

        if sum(β) == 0
            β[1] = sum(y)/length(y)
        end

        βtemp = similar(β)


        for i in 1:maxiter
            η .= X * β
            μ .= InverseLinkFunction(link, η)
            w = Var(dist, μ)

            U = Score(X, y, μ, dist, link)
            J = FisherInformation(X, w, dist, link)
            Jinv = inv(J)

            βtemp .= β

            β .= β + Jinv * U
            δ = β - βtemp
            if sqrt(δ' *  δ) < tol
                println(@sprintf "Stopping after %d iterations" i)
                M.Fit.SE = sqrt.(diag(Jinv))
                break
            else
                L = LogLikelihood(dist, μ, y)
                println(@sprintf "Fisher Iteration %d 𝐿(β) = %.12e" i L)
                M.Fit.SE = sqrt.(diag(Jinv))
            end
        end
        #M.Fit.β .= β
        η .= X * β
        μ .= InverseLinkFunction(link, η)
        M.Fit.𝐃 .= Deviance(dist, μ, y)
        M.Fit.DoF = length(y) - length(β)
        l = LogLikelihood(dist, μ, y)
        M.Fit.AIC = AIC(length(β), l)
        M.IsFitted = true
end
