struct DPG <: PolicyGradientAlgorithm
    # Hyperparameters
    Ntraj::Int # Number of trajectories
    Nsteps::Number # Max number of discrete time-steps before a trajectory is finished
    Nepisodes::Int # Number of episodes
    gamma::Number # Discount factor
    batchsize::Int # Number of trajectories used to compute one alternative loss

    # Parameters related to the Value function
    nH::Int # Dimension of the input into the advantage function. Should be env.nX for Markov environments.
    valfunc # Advantage function, should contain Knet.Params to train it
    atype::Type
    usegpu::Bool
    optimizer  # The optimizer for the Advantage function, e.g. Knet.Adam()
    valmethod::Symbol # Defines how the value function targets are computed. Can be :MC or :TD0.
    lambda::Number # For the generalised advantage estimation (gae), TD-lambda
end


function minimize!(rl::DPG, pol::Policy, env::Environment, options::Options)
    checkconsistency(rl, pol, env)
    cputype = eltype(pol.atype)
    costvec = [NaN for i = 1:rl.Nepisodes]

    # Define loss and batch generator
    lossfun(dpginput) = dpgloss!(pol, dpginput)

    # Compute gradient using backprop and apply gradient
    optiter = Knet.minimize(lossfun, dpgit, pol.optimizer) # Why does this run one episode already?
    next = iterate(optiter)
    oldloss = Inf
    newepcount = 0 # increases when the loss increases. Sample new trajectories after newepcount=20
    while next !== nothing
        (loss, state) = next
        @assert(!isnan(loss), "Loss turned out NaN")
        @assert(!isinf(loss), "Loss turned out Inf")
        next = iterate(optiter, state)
    end

    return costvec
end


function iterate(dpgit::DPGIterator)
    return dpginput
end


function dpgloss!(pol::Policy, dpginput)
end
