using ReinforcementLearning
@everywhere using Knet, RNN

# Load the environment "env"
include("../environments/mountaincar.jl")

# Create a static policy as a simple fully connected NN
atype = (Knet.gpu() == -1 ? Array{Float32} : KnetArray{Float32})
cputype = eltype(atype)
usegpu = (Knet.gpu() != -1)
umean = Chain( (Dense(2, 16; atype = atype, activation = tanh), Dense(16, 1, atype = atype, activation = identity) ) )

pol = StaticPolicy(2, 1, cputype(1.0), umean, atype, usegpu, rnnconvert, Knet.Adam())

# Set up the algorithm
valfunc = Chain( (Dense(2, 32; atype = atype, activation = tanh), Dense(32, 16; atype = atype, activation = tanh), Dense(16, 1, atype = atype, activation = identity) ) )
alg = PPO(1000, 500, 10, cputype(0.99), cputype(0.1), 200, 2, valfunc, atype, usegpu, Knet.Adam(), :MC, cputype(0.05))
options = Options(5, 200, "mc.jld2",  default_worker_pool())
# Run the algorithm
all_costs = minimize!(alg, pol, env, options)

# Take a look at the value function
using Plots
pyplot()
x = range(cputype(-1.2); stop = cputype(0.5), length = 30 )
v = range(cputype(-0.07); stop = cputype(0.07), length = 30 )
surface(x, v, (x, y)->value(valfunc(atype(vcat(x, y))))[1])

# Test the policy
X = Array{Float32}(undef, 2, alg.Nsteps)
X[:, 1] .= [-pi/6, 0.0]
U = Array{Float32}(undef, 1, alg.Nsteps)
for i = 2:alg.Nsteps
    U[:, i-1] .= pol(X[:, i-1])
    X[:, i] .= env.dynamics(X[:, i-1], U[:, i-1])[1]
end
fig = plot(layout=(2, 1), title = "mountain car")
plot!(fig[1], X[1, :], lab = "pos")
plot!(fig[2], U[1, :], lab = "u")
