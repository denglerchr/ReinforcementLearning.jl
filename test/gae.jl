using Test, ReinforcementLearning

valfunc(x) = ones(size(sum(x, dims=1)))
testalg = PPO(3, 5, 1, 0.5, 1, 2, valfunc, Array{Float64}, false, nothing, :MC, 1.0, 1, default_worker_pool())

# Create a value function, returning constant 1
X = ones(2, 3, 5)
r = ones(1, 3, 5)

# compute the gae
expectgae = (1+0.5)*(-1+1+0.5*1)*ones(size(r))
expectgae[:, :, end] .= 0

@test ReinforcementLearning.gae(testalg, X, r) == expectgae