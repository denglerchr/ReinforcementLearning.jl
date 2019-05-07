@everywhere using ReinforcementLearning, Knet, RNN

# Load the environment "env"
include("../environments/pendulumv0_nonMarkov.jl")

# Create a static policy as a simple fully connected NN
atype = (Knet.gpu() == -1 ? Array{Float32} : KnetArray{Float32})
usegpu = (Knet.gpu() != -1)
@everywhere struct UMEAN # wrapper for returning also the hidden state
    nn::Chain
end
@everywhere (u::UMEAN)(X) = (U = u.nn(X); return U, vec( u.nn.layers[1].h) )

function convertumean(umeanin)
    nn = RNN.rnnconvert(umeanin.nn; atype = Array{Float64})
    return UMEAN(nn)
end

umean = UMEAN( Chain( (Knet.RNN(1, 16; rnnType = :gru, usegpu = usegpu, dataType = eltype(atype), h=0), Dense(16, 1, atype = atype, activation = identity) ) ) )
pol = RecurrentPolicy(1, 16, 1, 0.02, umean, atype, usegpu, convertumean , Knet.SGD(;lr=0.001), 100, rnn->hiddentozero!(rnn.nn))


# Define a value function, we use a neural network again
valfunc = Chain( (Dense(16, 64; atype = atype, activation = tanh), Dense(64, 32; atype = atype, activation = tanh), Dense(32, 1, atype = atype, activation = identity) ) )

# Set up the algorithm
alg = PPO(1000, 500, 500, 0.99, 0.05, 512, 16, valfunc, atype, usegpu, Knet.SGD(), :MC, 0.05, 2, default_worker_pool())

# Start training
all_costs = minimize!(alg, pol, env)



if usegpu
    pol2 = ReinforcementLearning.cpupol(pol)
else
    pol2 = pol
end

ReinforcementLearning.resetpol!(pol2)
X = Array{Float64}(undef, 1, alg.Nsteps)
X[:, 1] .= env.resetenv!(env.dynamics)
U = Array{Float64}(undef, 1, alg.Nsteps)
for i = 2:alg.Nsteps
    U[:, i-1] .= pol2(X[:, i-1])[1]
    X[:, i] .= env.dynamics(X[:, i-1], U[:, i-1])[1]
end
using Plots
unicodeplots()
fig = plot(layout=(2, 1), title = "pendulumv0")
plot!(fig[1], X[1, :], lab = "angle")
plot!(fig[2], U[1, :], lab = "u")
