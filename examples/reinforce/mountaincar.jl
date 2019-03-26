@everywhere using ReinforcementLearning, Knet, RNN

# Load the environment "env"
include("../environments/mountaincar.jl")

# Create a static policy as a simple fully connected NN
atype = Array{Float64}
umean = Chain( (Dense(2, 16; atype = atype, activation = tanh), Dense(16, 1, atype = atype, activation = identity) ) )

pol = StaticPolicy(2, 1, 1.0, x->umean(x), atype, false, nothing, Knet.Adam())

# Set up the algorithm
alg = Reinforce(10, 500, 500, 1.0, :mean, 50, default_worker_pool())

# Run the algorithm
all_costs = minimize!(alg, pol, env)

# Test the policy
X = Array{Float64}(undef, 2, alg.Nsteps)
X[:, 1] .= [-pi/6, 0.0]
U = Array{Float64}(undef, 1, alg.Nsteps)
for i = 2:alg.Nsteps
    U[:, i-1] .= pol(X[:, i-1])
    X[:, i] .= env.dynamics(X[:, i-1], U[:, i-1])[1]
end
using Plots
fig = plot(layout=(2, 1), title = "mountain car")
plot!(fig[1], X[1, :], lab = "pos")
plot!(fig[2], U[1, :], lab = "u")