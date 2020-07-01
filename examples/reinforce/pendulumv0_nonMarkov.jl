@everywhere using ReinforcementLearning, Knet, RNN

# Load the environment "env"
include("../environments/pendulumv0_nonMarkov.jl")

# Create a static policy as a simple fully connected NN
atype = Array{Float64}
usegpu = false
@everywhere struct UMEAN # wrapper for returning also the hidden state
    nn::Chain
end
@everywhere (u::UMEAN)(X) = (U = u.nn(X); return U, vec( u.nn.layers[1].h) )

umean = UMEAN( Chain( (Knet.RNN(1, 16; rnnType = :gru, usegpu = usegpu, dataType = eltype(atype), h=0), Dense(16, 1, atype = atype, activation = identity) ) ) )
pol = RecurrentPolicy(1, 16, 1, 0.2, umean, atype, usegpu, nothing, Knet.Adam(), 100, rnn->hiddentozero!(rnn.nn))

# Set up the algorithm
alg = Reinforce(400, 300, 500, 0.99, :mean)
options = Options(10, 100, "pend_nm.jld2",  default_worker_pool())

# Run the algorithm
all_costs = minimize!(alg, pol, env, options)

# Test the policy
X = Array{Float64}(undef, 1, alg.Nsteps)
X[:, 1] .= env.resetenv!(env.dynamics)
U = Array{Float64}(undef, 1, alg.Nsteps)
ReinforcementLearning.resetpol!(pol)
for i = 2:alg.Nsteps
    U[:, i-1] .= pol(X[:, i-1])
    X[:, i] .= env.dynamics(X[:, i-1], U[:, i-1])[1]
end
using Plots
fig = plot(layout=(2, 1), title = "pendulumv0")
plot!(fig[1], X[1, :], lab = "angle")
plot!(fig[2], U[1, :], lab = "u")
