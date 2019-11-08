module GeneralizedLinearModels

    using LinearAlgebra, Printf
    import Distributions: NormalCanon, ccdf


    export
        # Exponential Family Distributions
        Binomial,

        # Links
        Logit,

        # Models
        LinearPredictor,
        GLModel,
        GLMFit,

        # Functions
        Score,
        FisherInformation,
        FisherScoring!,
        LinkFunction,
        InverseLinkFunction,
        CanonicalLink,
        Var,
        LogLikelihood,
        Likelihood,
        Deviance,
        glm,

        # Utils
        logistic,
        logit,

        #StatsUtils
        summary



    abstract type AbstractGLM end

    abstract type ExponentialFamily end
    abstract type AbstractLinearPredictor end
    abstract type Link end

    include("Utils.jl")
    include("Binomial.jl")
    include("Links.jl")
    include("LinearPredictor.jl")
    include("Model.jl")
    include("Fit.jl")
    include("FisherScoring.jl")
    include("StatsUtils.jl")
end
