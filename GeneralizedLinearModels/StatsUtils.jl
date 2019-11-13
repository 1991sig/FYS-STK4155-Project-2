
function summary(M::GLMFit)
    β = M.Fit.β
    colnames = M.Model.cols
    SE = M.Fit.SE

    @printf("\n")

    t1 = rpad("Variable",15," ")
    t2 = rpad("β-hat", 14, " ")
    t3 = rpad("SE",13," ")
    t4 = rpad("Z",8," ")
    t5 = rpad("P(z > |Z|)",15," ")

    @printf "%s %s %s %s %s" t1 t2 t3 t4 t5
    @printf("\n")

    t1 = rpad("--------",15," ")
    t2 = rpad("-----", 14, " ")
    t3 = rpad("--",13," ")
    t4 = rpad("-",8," ")
    t5 = rpad("----------",15," ")

    @printf "%s %s %s %s %s" t1 t2 t3 t4 t5
    @printf("\n")

    for i in eachindex(β, SE, colnames)
        v1 = rpad(colnames[i],12," ")
        v2 = β[i]
        v3 = SE[i]
        v4 = v2/v3
        v5 = 2*ccdf(NormalCanon(), abs(v4))
        @printf "%s % .5e % .5e % .5e   % .5f " v1 v2 v3 v4 v5
        @printf("\n")
    end

    @printf("\n")
    @printf("\n")
    @printf "AIC: %d" M.Fit.AIC
    @printf("\n")
    @printf "DoF: %d" M.Fit.DoF
    @printf("\n")
    @printf "Residual Deviance: %d" sum(M.Fit.𝐃)

end
