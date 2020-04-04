using Test, ReinforcementLearning

r = ones(1, 3, 3)
gamma = 0.5
rdiscounted = reshape([1, 1, 1, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25], size(r))
@test ReinforcementLearning.applydiscount!(r, gamma) == rdisocunted