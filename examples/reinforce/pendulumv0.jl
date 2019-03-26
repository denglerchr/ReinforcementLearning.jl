using ReinforcementLearning, Knet, RNN

# Load the environment "env"
include("../environments/pendulumv0.jl")

# Create a static policy as a simple fully connected NN
atype = Array{Float64}
umean = Chain( (Dense(2, 32; atype = atype, activation = tanh), Dense(32, 16; atype = atype, activation = tanh), Dense(16, 1, atype = atype, activation = identity) ) )

pol = StaticPolicy(2, 1, 0.2, x->umean(x), atype, false, nothing, Knet.Adam())

# Set up the algorithm
alg = Reinforce(400, 500, 1000, 0.99, :mean, 50, default_worker_pool())

# Run the algorithm
all_costs = minimize!(alg, pol, env)

# Test the policy
X = Array{Float64}(undef, 2, alg.Nsteps)
X[:, 1] .= [-pi, 0.0]
U = Array{Float64}(undef, 1, alg.Nsteps)
for i = 2:alg.Nsteps
    U[:, i-1] .= pol(X[:, i-1])
    X[:, i] .= env.dynamics(X[:, i-1], U[:, i-1])[1]
end
using Plots
fig = plot(layout=(2, 1), title = "pendulumv0")
plot!(fig[1], X[1, :], lab = "angle")
plot!(fig[1], X[2, :], lab = "vel")
plot!(fig[2], U[1, :], lab = "u")