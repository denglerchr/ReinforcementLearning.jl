# Define types to distinguish between different types of RL Algorithms
abstract type RLAlgorithm end
abstract type PolicyGradientAlgorithm<:RLAlgorithm end


## Functions used in many algorithms
#meanU and U should be 3d Tensors
function pdfgaussian(meanU, U, std::Number)
    return exp.(logpdfgaussian(meanU, U, std))
end


function logpdfgaussian(meanU, U, std::Number)
    @assert(size(meanU) == size(U))

    # Number of variables in the distribution
    N = size(U, 1)

    # The normalizing factor of the distribution, after applying the log
    c = -log(pi*std^2*2)*N/2

    # Mean term (3d tensor)
    Temp = sum( abs2.(U.-meanU) , dims=1)
    logpdf = c .- Temp ./std^2 ./2

    return logpdf
end


# Return p1(U)/p2(U) for Gaussian distributions
function pdfgaussianquot(meanU1, meanU2, U, std1, std2)
    return exp.(logpdfgaussian(meanU1, U, std1) .- logpdfgaussian(meanU2, U, std2))
end

# Return p1(U)/p2(U) for Gaussian distributions, but cut off to max_val (to avoid Inf)
function pdfgaussianquot_limited(meanU1, meanU2, U, std1::Number, std2::Number, max_val::Number)
    logpdf = min.( log(max_val), logpdfgaussian(meanU1, U, std1) .- logpdfgaussian(meanU2, U, std2) )
    return exp.(logpdf)
end
