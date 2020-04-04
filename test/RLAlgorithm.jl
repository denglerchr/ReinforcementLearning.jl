using ReinforcementLearning, Distributions, Test

# Create random testcase
umeantest = repeat(randn(5, 200), outer=(1, 1, 300))
umeantest2 = repeat(randn(5, 200), outer=(1, 1, 300))
utest = repeat(randn(5, 200), outer = (1, 1, 300))
std = randn()^2 + 0.5
std2 = randn()^2 + 0.5

# Create hopefully correct values using Distributions.jl
function pdfgaussian2(Umean, U, std::Number)
    @assert size(Umean) == size(U)
    dist = Distributions.MvNormal(size(U, 1), std)
    Out = similar(U, (1, size(U, 2), size(U, 3)))
    for j = 1:size(U, 3), i = 1:size(U, 2)
        Out[1, i, j] = Distributions.pdf(dist, Umean[:, i, j]-U[:, i, j])
    end
    return Out
end
function logpdfgaussian2(Umean, U, std::Number)
    @assert size(Umean) == size(U)
    dist = Distributions.MvNormal(size(U, 1), std)
    Out = similar(U, (1, size(U, 2), size(U, 3)))
    for j = 1:size(U, 3), i = 1:size(U, 2)
        Out[1, i, j] = Distributions.logpdf(dist, Umean[:, i, j]-U[:, i, j])
    end
    return Out
end
function pdfgaussianquot2(meanU1, meanU2, U, std1, std2)
    return exp.(logpdfgaussian2(meanU1, U, std1) .- logpdfgaussian2(meanU2, U, std2))
end


# Compare with implementation
println("Testing")
@test ReinforcementLearning.pdfgaussian(umeantest, utest, std) ≈ pdfgaussian2(umeantest, utest, std)
@test ReinforcementLearning.logpdfgaussian(umeantest, utest, std) ≈ logpdfgaussian2(umeantest, utest, std)
@test ReinforcementLearning.pdfgaussianquot(umeantest, umeantest2, utest, std, std2) ≈ pdfgaussianquot2(umeantest, umeantest2, utest, std, std2)