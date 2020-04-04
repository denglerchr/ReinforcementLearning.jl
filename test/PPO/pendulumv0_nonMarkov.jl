using ReinforcementLearning, Plots, Knet
@everywhere using RNN
# Load the pendulum problem

include("../../examples/environments/pendulumv0_nonMarkov.jl")
atype = Array{Float32}#(Knet.gpu() == -1 ? Array{Float32} : KnetArray{Float32})
cputype = eltype(atype)
usegpu = false#(Knet.gpu() != -1)

@everywhere struct UMEAN # wrapper for returning also the hidden state
    nn::Chain
end
@everywhere (u::UMEAN)(X) = (U = u.nn(X); return U, vec( u.nn.layers[1].h) )

umean = UMEAN( Chain( (Knet.RNN(1, 16; rnnType = :gru, usegpu = usegpu, dataType = eltype(atype), h=0), Dense(16, 1, atype = atype, activation = identity) ) ) )
pol = RecurrentPolicy(1, 16, 1, 0.2, umean, atype, usegpu, nothing, Knet.Adam(), 100, rnn->hiddentozero!(rnn.nn))


# Define a value function, we use a neural network again
valfunc = Chain( (Dense(16, 32; atype = atype, activation = tanh), Dense(32, 16; atype = atype, activation = tanh), Dense(16, 1, atype = atype, activation = identity) ) )

# Set up the algorithm
rl = PPO(50, 500, 2, 0.99, 0.05, 50, 16, valfunc, atype, usegpu, Knet.Adam(), :MC, 0.3, 2, default_worker_pool())

ReinforcementLearning.checkconsistency(rl, pol, env)

# Create and plot some trajectories
costvec = zeros(rl.Nepisodes)
ppodata = ReinforcementLearning.PPOdata(zeros(cputype, 1, 1, 1), zeros(cputype, 1, 1, 1), zeros(cputype, 1, 1, 1), zeros(cputype, 1, 1, 1))
ppoit = ReinforcementLearning.PPOIterator(env, pol, rl, ppodata, costvec, [true])
iterate(ppoit)

sometraj = ReinforcementLearning.gettrajh(pol, env, rl)

plotind = rand(1:length(sometraj))
#=
fig = plot(layout = (3, 1))
plot!(fig[1], sometraj[plotind].X')
plot!(fig[2], sometraj[plotind].H')
plot!(fig[3], sometraj[plotind].U')
=#

# Stack them into big tensors and reproduce the first plot
X, H, U, r = ReinforcementLearning.stackXHUr(sometraj, pol, pol.seqlength)

#=
fig = plot(layout = (3, 1))
plot!(fig[1], X[:, plotind, :]')
plot!(fig[2], H[:, plotind, :]')
plot!(fig[3], U[:, plotind, :]')
=#

# train the Value function and have a look at the value targets
ReinforcementLearning.valuetrain!(rl, H, r)
vtarget = ReinforcementLearning.valuetarget(r, rl, H, Val(:MC))
vtarget = ReinforcementLearning.valuetarget(r, rl, H, Val(:TD0))

# Generalised advantage estimation
A = ReinforcementLearning.gae(rl, H, r)
A = rl.atype(A)

# Compute the mean of the old policy
ReinforcementLearning.resetpol!(pol)
meanUold = pol.umean(pol.atype(X))[1]
#=
plot(U[:, plotind, :]')
plot!(meanUold[:, plotind, :]')
=#

ppodata = ReinforcementLearning.PPOdata(X, U, A)
lossinput = ReinforcementLearning.getbatch(ppodata, 10, pol)

# The same for now, would change after applying a gradient
meanUnew = meanUold
probquot = ReinforcementLearning.pdfgaussianquot(meanUnew, meanUold, rl.atype(U), pol.std, pol.std)

# Clipping
epsilon = 0.05
clipbool = (A .> 0) # 1 where probquot should be clipped at 1-epsilon, 0 if clipping at 1+epsilon
probquotclipped = clipbool.*max.(1.0-epsilon, probquot) .+ (1 .- clipbool) .* min.(1.0+epsilon, probquot)

# return the mean of the product, which is what is minimized here (usually called "L")
probquotclipped.*A

ReinforcementLearning.ppoloss!(pol, epsilon, lossinput)
