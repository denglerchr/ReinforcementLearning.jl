@everywhere using ReinforcementLearning, Knet, RNN

# Load the environment "env"
include("../environments/chain.jl")

# Create a static policy as a simple fully connected NN
atype = Array{Float64}
 
@everywhere struct UMEAN # wrapper for returning also the hidden state
    nn::Chain
end
@everywhere (u::UMEAN)(X) = (U = u.nn(X); return U, u.nn.layers[1].h)

umean = UMEAN( Chain( (Knet.RNN(2, 32; rnnType = :gru, usegpu = false, dataType = eltype(atype), h=0), Dense(32, 1, atype = atype, activation = identity) ) ) )
#umean = Chain( (GRU(2, 16; atype = atype), Dense(16, 1, atype = atype, activation = identity) ) )
pol = RecurrentPolicy(2, 32, 1, 0.2, umean, atype, false, nothing, Knet.Adam(), rnn->hiddentozero!(rnn.nn))

# Set up the algorithm
alg = Reinforce(400, 300, 500, 0.995, :mean, 1, default_worker_pool())

# Run the algorithm
all_costs = minimize!(alg, pol, env)

# Test the policy
X = Array{Float64}(undef, 2, alg.Nsteps)
X[:, 1] .= env.resetenv!(env.dynamics)
U = Array{Float64}(undef, 1, alg.Nsteps)
ReinforcementLearning.resetpol!(pol)
for i = 2:alg.Nsteps
    U[:, i-1] .= pol(X[:, i-1])
    X[:, i] .= env.dynamics(X[:, i-1], U[:, i-1])[1]
end
using Plots
fig = plot(layout=(2, 1), title = "HeavyChain")
plot!(fig[1], X[1, :], lab = "pos")
plot!(fig[2], U[1, :], lab = "u")


# profile things
function test(Nsteps, env, pol)
    X = Array{Float64}(undef, 2, Nsteps)
    X[:, 1] .= env.resetenv!(env.dynamics)
    U = Array{Float64}(undef, 1, Nsteps)
    ReinforcementLearning.resetpol!(pol)
    for i = 2:Nsteps
        U[:, i-1] .= pol(X[:, i-1])
        X[:, i] .= env.dynamics(X[:, i-1], U[:, i-1])[1]
    end
    return X, U
end