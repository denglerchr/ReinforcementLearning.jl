using ReinforcementLearning
@everywhere using Knet, RNN

# Load the environment "env"
include("../../examples/environments/mountaincar.jl")

# Create a static policy as a simple fully connected NN
atype = (Knet.gpu() == -1 ? Array{Float32} : KnetArray{Float32})
cputype = eltype(atype)
usegpu = (Knet.gpu() != -1)
umean = Chain( (Dense(2, 16; atype = atype, activation = tanh), Dense(16, 1, atype = atype, activation = identity) ) )

pol = StaticPolicy(2, 1, 0.5, umean, atype, usegpu, rnnconvert, Knet.Adam())

# Set up the algorithm
valfunc = Chain( (Dense(2, 32; atype = atype, activation = tanh), Dense(32, 16; atype = atype, activation = tanh), Dense(16, 1, atype = atype, activation = identity) ) )
alg = PPO(1000, 500, 10, Float32(0.99), Float32(0.05), 200, 2, valfunc, atype, usegpu, Knet.Adam(), :MC, Float32(0.05), 2, default_worker_pool())


# Create and plot some trajectories
sometraj = ReinforcementLearning.gettraj(pol, env, alg)

plotind = rand(1:length(sometraj))
#=
fig = plot(layout = (2, 1))
plot!(fig[1], sometraj[plotind].X')
plot!(fig[2], sometraj[plotind].U')
=#

# Stack them into big tensors and reproduce the first plot
X, U, r = ReinforcementLearning.stackXUr(sometraj, pol)

#=
fig = plot(layout = (3, 1))
plot!(fig[1], X[:, plotind, :]')
plot!(fig[2], U[:, plotind, :]')
plot!(fig[3], r[:, plotind, :]')
=#

# train the Value function and have a look at the value targets
#ReinforcementLearning.valuetrain!(alg, X, r)
vtarget = ReinforcementLearning.valuetarget(r, alg, X, Val(:TD0))
vtarget = ReinforcementLearning.valuetarget(r, alg, X, Val(:MC))

X2 = reshape(X, size(X, 1), size(X, 2)*size(X, 3))
vtarget2 = reshape(vtarget, size(vtarget, 1), size(vtarget, 2)*size(vtarget, 3))
indices = randperm(size(X2, 2))
trainindices = indices[1:ceil(Int, length(indices)*0.8)]
testindices = indices[ceil(Int, length(indices)*0.8)+1:end]
Xtrain = alg.atype( X2[:, trainindices] )
Xtest = alg.atype( X2[:, testindices] )
Ytrain = alg.atype( vtarget2[:, trainindices] )
Ytest = alg.atype( vtarget2[:, testindices] )

batchsize = max( ceil(Int, size(Xtrain, 2)/10), min( size(Xtrain, 2), 512 ) ) #TODO should this be a parameter of the algorithm, or is a standard value ok
data = Knet.minibatch(Xtrain, Ytrain, batchsize; shuffle = true)

scatter(Array{cputype}(Xtrain[1, 1:300]), Array{cputype}(Xtrain[2, 1:300]), Array{cputype}(vec(valfunc(Xtrain))[1:300]), xlim = (-1.2, 0.5), ylim = (-0.07, 0.07), xlab = "x", ylab = "v")
scatter(Array{cputype}(Xtrain[1, 1:300]), Array{cputype}(Xtrain[2, 1:300]), Array{cputype}(Ytrain[1, 1:300]), xlim = (-1.2, 0.5), ylim = (-0.07, 0.07), xlab = "x", ylab = "v")

# Generalised advantage estimation
A = ReinforcementLearning.gae(alg, X, r)
#=
fig = plot(layout = (4, 1))
plot!(fig[1], X[:, 1, :]')
plot!(fig[2], U[:, 1, :]')
plot!(fig[3], r[:, 1, :]')
plot!(fig[4], A[:, 1, :]')
=#
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
