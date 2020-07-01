using ReinforcementLearning, Knet, RNN

# Load the environment "env"
include("../environments/pendwithcart.jl")

# Create a static policy as a simple fully connected NN
usegpu = false#(Knet.gpu() != -1)
atype = (usegpu ? KnetArray{Float32} : Array{Float32})
umean = Chain( (Dense(4, 16; atype = atype, activation = tanh), Dense(16, 8; atype = atype, activation = tanh), Dense(8, 1, atype = atype, activation = identity) ) )

pol = StaticPolicy(4, 1, Float32(1.0), umean, atype, true, x->RNN.rnnconvert(x; atype = Array{Float32}), Knet.Adam())

# Define a value function, we use a neural network again
valfunc = Chain( (Dense(4, 16; atype = atype, activation = tanh), Dense(16, 8; atype = atype, activation = tanh), Dense(8, 1, atype = atype, activation = identity) ) )

# Set up the algorithm
alg = PPO(500, 100, 300, Float32(0.999), Float32(0.1), 200, 4, valfunc, atype, true, Knet.Adam(), :TD0, Float32(0.5))
options = Options(2, -1, "pend.jld2",  default_worker_pool())

# Run the algorithm
all_costs = minimize!(alg, pol, env, options)
Knet.@save("pendwithcart.jld2", alg, pol)

# Test the policy
if usegpu
    pol2 = ReinforcementLearning.cpupol(pol)
else
    pol2 = pol
end

X = Array{Float32}(undef, 4, alg.Nsteps)
X[:, 1] .= [-0.5, 0.0, 0.0, 0.0]
U = Array{Float32}(undef, 1, alg.Nsteps)
for i = 2:alg.Nsteps
    U[:, i-1] .= pol2(X[:, i-1])
    X[:, i] .= env.dynamics(X[:, i-1], U[:, i-1])[1]
end
using Plots
fig = plot(layout=(2, 1), title = "pendulum with cart")
plot!(fig[1], X[1, :], lab = "pos")
plot!(fig[1], X[2, :], lab = "vel")
plot!(fig[1], X[3, :], lab = "alpha")
plot!(fig[1], X[4, :], lab = "dalpha")
plot!(fig[2], U[1, :], lab = "u")
