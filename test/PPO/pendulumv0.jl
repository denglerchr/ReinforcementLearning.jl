# This does not produce numerical tests, but goes through the algorithm step by step and plot things to check by a user
using ReinforcementLearning, Plots, RNN, Knet

# Load the pendulum problem

include("../../examples/environments/pendulumv0.jl")
atype = (Knet.gpu() == -1 ? Array{Float32} : KnetArray{Float32})
usegpu = (Knet.gpu() != -1)

umean = Chain( (Dense(2, 16; atype = atype, activation = tanh), Dense(16, 16; atype = atype, activation = tanh), Dense(16, 1, atype = atype, activation = identity) ) )

pol = StaticPolicy(2, 1, 0.3, umean, atype, usegpu, x->RNN.rnnconvert(x; atype = Array{Float32}), Knet.Adam())

valfunc = Chain( (Dense(2, 16; atype = atype, activation = tanh), Dense(16, 16; atype = atype, activation = tanh), Dense(16, 1, atype = atype, activation = identity) ) )

alg = PPO(200, 500, 10, 0.99, 0.05, 200, 2, valfunc, atype, usegpu, Knet.Adam(), :MC, 0.3, 50, default_worker_pool())

# Create and plot some trajectories
sometraj = ReinforcementLearning.gettraj(pol, env, alg)

plotind = rand(1:length(sometraj))
#=
fig = plot(layout = (2, 1))
plot!(fig[1], sometraj[plotind].X')
plot!(fig[2], sometraj[plotind].U')
=#

# Stack them into big tensors and reproduce the first plot
X, U, r = ReinforcementLearning.stacktraj(sometraj, pol)

#=
fig = plot(layout = (2, 1))
plot!(fig[1], X[:, plotind, :]')
plot!(fig[2], U[:, plotind, :]')
=#

# train the Value function and have a look at the value targets
ReinforcementLearning.valuetrain!(alg, X, r)
vtarget = ReinforcementLearning.valuetarget(r, alg, X, Val(:MC))
vtarget = ReinforcementLearning.valuetarget(r, alg, X, Val(:TD0))

# Generalised advantage estimation
A = ReinforcementLearning.gae(alg, X, r)
A = alg.atype(A)

ppodata = ReinforcementLearning.PPOdata(X, U, A)
lossinput = ReinforcementLearning.getbatch(ppodata, 50, pol)
#=
fig = plot(layout = (2, 1))
plot!(fig[1], lossinput.X[:, 1, :]')
plot!(fig[2], lossinput.U[:, 1, :]')
=#

# Compute the mean of the old policy
ReinforcementLearning.resetpol!(pol)
meanUold = ReinforcementLearning.umean1(pol, pol.atype(X))
#=
plot(U[:, plotind, :]')
plot!(meanUold[:, plotind, :]')
=#

# The same for now, would change after applying a gradient
meanUnew = meanUold
probquot = ReinforcementLearning.pdfgaussianquot(meanUnew, meanUold, alg.atype(U), pol.std, pol.std)

# Clipping
epsilon = 0.05
clipbool = (A .> 0) # 1 where probquot should be clipped at 1-epsilon, 0 if clipping at 1+epsilon
probquotclipped = clipbool.*max.(1.0-epsilon, probquot) .+ (1 .- clipbool) .* min.(1.0+epsilon, probquot)

# return the mean of the product, which is what is minimized here (usually called "L")
probquotclipped.*A

ReinforcementLearning.ppoloss!(pol, epsilon, lossinput)
