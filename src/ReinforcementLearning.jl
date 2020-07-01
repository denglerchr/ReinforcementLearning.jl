# Provide standard reinforcement learning algorithms for (until now) continuous state and action spaces.
# The code is based on the Knet and AutoGrad module for computing the gradients.
# The main structs are:
# 1. The "Environment" struct, see ?Environment
# 2. The "Policy" struct. Some algorithms also accept recurrent policies or GPU training.
# 3. The struct defining the algorithm and its hyperparameters
#   3.1 "Reinforce" struct for the reinforce algorithm
#   3.2 "PPO" struct for the PPO algorithm
#
# If all those structs are defined, you can start optimizing by calling
# optimize!(algo, policy, environment), which returns the average
module ReinforcementLearning

using Knet, Statistics, Distributed, Random
import Base.iterate, Base.length, Knet.minimize!

include("Environment.jl")
export Environment

include("Policy.jl")
export StaticPolicy, RecurrentPolicy

#
include("RLAlgorithm.jl")
include("Trajectory.jl")
include("options.jl")
export Options
include("algorithms/Reinforce.jl")
export Reinforce
export minimize!

include("algorithms/PPO.jl")
include("valuefunc.jl")
include("gae.jl")
export PPO

include("algorithms/PPOEpopt.jl")
export PPOEpopt

include("algorithms/Experimental.jl")
export Experimental

end
