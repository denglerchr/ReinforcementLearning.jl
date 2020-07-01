struct PPOEpopt <: PolicyGradientAlgorithm
    # Hyperparameters
    Ntraj::Int # Number of trajectories
    Nsteps::Number # Max number of discrete time-steps before a trajectory is finished
    Nepisodes::Int # Number of episodes
    gamma::Number # Discount factor
    epsilon::Number # Clipping factor for PPO
    batchsize::Int # Number of trajectories used to compute one alternative loss

    # Parameters related to the Value function
    nH::Int # Dimension of the input into the advantage function. Should be env.nX for Markov environments.
    valfunc # Advantage function, should contain Knet.Params to train it
    atype::Type
    usegpu::Bool
    optimizer  # The optimizer for the Advantage function, e.g. Knet.Adam()
    valmethod::Symbol # Defines how the value function targets are computed. Can be :MC or :TD0.
    lambda::Number # For the generalised advantage estimation (gae), TD-lambda

    # Parameters related to Epopt
    e_perc::Number # epsilon percentile of worst trajectories used
end

# Contains trajectory data as tensors (always on the cpu)
mutable struct PPOEpoptData{T}
    X::T
    U::T
    meanUold::T
    A::T
    # Vector containing the indices of trajectories
    # sorted according to the trajectory cost. Starting at worst traj index
    sort_ind::Vector
end

struct PPOEpoptIterator{P<:Policy}
    env::Environment
    pol::P
    rl::PPOEpopt
    alldata::PPOEpoptData # save all data of one iteration
    costvec::Vector{Float64}
    newepisode::Vector{Bool}
    printevery::Int # Print loss every xx episodes
    workerpool::WorkerPool
end

# The main call, starting the rl optimization
function minimize!(rl::PPOEpopt, pol::Policy, env::Environment, options::Options = Options())
    checkconsistency(rl, pol, env, options)
    cputype = eltype(pol.atype)
    costvec = [NaN for i = 1:rl.Nepisodes]

    # Define loss and batch generator
    lossfun(ppoinput) = ppoloss!(pol, rl.epsilon, ppoinput)
    ppodata = PPOEpoptData(zeros(cputype, 1, 1, 1), zeros(cputype, 1, 1, 1), zeros(cputype, 1, 1, 1), zeros(cputype, 1, 1, 1), zeros(Int, rl.Ntraj))
    ppoit = PPOEpoptIterator(env, pol, rl, ppodata, costvec, [true], options.printevery, options.workerpool)

    # Compute gradient using backprop and apply gradient
    optiter = Knet.minimize(lossfun, ppoit, pol.optimizer) # Why does this run one episode already?
    next = iterate(optiter)
    oldloss = Inf
    newepcount = 0 # increases when the loss increases. Sample new trajectories after newepcount=20
    while next !== nothing
        (loss, state) = next
        @assert(!isnan(loss), "Loss turned out NaN")
        @assert(!isinf(loss), "Loss turned out Inf")
        (state[2] == 1) && Options_savepol(options, pol, costvec, state[1]) # Possibly save policy
        if (loss>=oldloss)
            newepcount += 2
            newepcount>30 && (ppoit.newepisode[1] = true; newepcount = 0; oldloss=Inf)
        else
            oldloss = loss
            newepcount = max(0, newepcount-1)
        end

        next = iterate(optiter, state)
    end

    return ppoit.costvec
end

function checkconsistency(rl::PPOEpopt, pol::Policy, env::Environment, options::Options)
    @assert pol.nX == env.nX
    @assert pol.nU == env.nU
    @assert pol.usegpu == rl.usegpu
    @assert pol.atype == rl.atype
    @assert rl.batchsize <= floor(Int, rl.Ntraj*rl.e_perc)
    if isa(pol, RecurrentPolicy)
        @assert pol.seqlength <= rl.Nsteps
        mod(rl.Nsteps, pol.seqlength) != 0 && @warn("Sequence length for the gradient is not a multiple of the number of timesteps. Last timesteps will not be used.")
    end

    # Check if filename is ok
    isfile(options.filename) && error("File $(options.filename) already exists, please delete or move it")
    return 0
end


## Functions for a static policy
# First iterate call, producting data that is fed into the forward pass
function iterate(ppoit::PPOEpoptIterator{<:P}) where {P<:Policy}
    getnewdata!(ppoit, 1)
    lossinput = getbatch(ppoit.alldata, ppoit.rl.e_perc, ppoit.rl.batchsize, ppoit.pol)
    return (lossinput,), (1, 1)
end

# first state is the episode number, second state is a counter for ppo, reusing old data to produce yet another gradient
function iterate(ppoit::PPOEpoptIterator{<:P}, state::Tuple{Int, Int}) where {P<:Policy}

    episN = state[1]
    incount = state[2]
    # check if we should get new data
    if (ppoit.newepisode[1] == false && incount < 1000)
        print("Epis. $episN , $(incount+1)\u1b[K\r")
        lossinput = getbatch(ppoit.alldata, ppoit.rl.e_perc, ppoit.rl.batchsize, ppoit.pol)
        return (lossinput,), (episN, incount+1)
    end

    ## Pass on to next episode, get new data
    newepisN = episN + 1
    newepisN > ppoit.rl.Nepisodes && return nothing

    getnewdata!(ppoit, newepisN)
    lossinput = getbatch(ppoit.alldata, ppoit.rl.e_perc, ppoit.rl.batchsize, ppoit.pol)

    return (lossinput,), (newepisN, 1)
end


# Sample new trajectories, train value function etc etc
function getnewdata!(ppoit::PPOEpoptIterator{<:StaticPolicy}, episN::Int)
        # Get trajectories
        print("Epis. $episN , 1 \u1b[K")
        trajvec = gettraj(ppoit.pol, ppoit.env, ppoit.rl, ppoit.workerpool)

        # Transform the data into big tensors for faster GPU computation
        X, U, r = stackXUr(trajvec, ppoit.pol)
        ppoit.alldata.X = X
        ppoit.alldata.U = U

        # Sort indices according to the mean costs
        sumvec = vec(mean(r; dims = 3))
        sortperm!(ppoit.alldata.sort_ind, sumvec; order=Base.Order.ReverseOrdering())

        # Recompute the mean of the policy, given this input
        ppoit.alldata.meanUold = typeof(U)( umean1(ppoit.pol, ppoit.pol.atype(X)) )

        # Train the value function
        valuetrain!(ppoit.rl, X, r)
        print("Epis. $episN , 1 \u1b[K\r")

        # Compute the "generalised advantage estimation"
        A = gae(ppoit.rl, X, r)
        ppoit.alldata.A = A

        # Carefull, overwriting r here
        applydiscount!(r, ppoit.rl.gamma)
        meancosts = mean(sum(r, dims = 3))
        ppoit.costvec[episN] = meancosts
        (mod(episN, ppoit.printevery) == 0 || episN == 1) && println("Epis. $episN : mean costs $(meancosts) \u1b[K") # \u1b[K cleans the rest of the line

        # Set other variables
        ppoit.newepisode[1] = false

        return 1
end

# Sample new trajectories, train value function etc etc (version for Recurrent Policies)
function getnewdata!(ppoit::PPOEpoptIterator{<:RecurrentPolicy}, episN::Int)
    # Get trajectories
    print("Epis. $episN , 1 \u1b[K")
    trajvec = gettrajh(ppoit.pol, ppoit.env, ppoit.rl, ppoit.workerpool)

    # Transform the data into big tensors for faster GPU computation
    # Also cut X and U in shorter sequences
    # Leave H and r in long sequences, to have correct computation of the advantage
    X, U = stackXU(trajvec, ppoit.pol, ppoit.pol.seqlength)
    H, r = stackHr(trajvec, ppoit.pol)
    ppoit.alldata.X = X
    ppoit.alldata.U = U

    # Sort indices according to the mean costs
    sumvec = vec(mean(r; dims = 3))
    sortperm!(ppoit.alldata.sort_ind, sumvec; order=Base.Order.ReverseOrdering())

    # Recompute the mean of the policy, given this input (can be different from simulation because of seqlength and reset)
    resetpol!(ppoit.pol)
    ppoit.alldata.meanUold = typeof(U)( umean1(ppoit.pol, ppoit.pol.atype(X)) )

    # Train the value function
    valuetrain!(ppoit.rl, H, r)
    print("Epis. $episN , 1 \u1b[K\r")

    # Compute the "generalised advantage estimation"
    A = gae(ppoit.rl, H, r)
    ppoit.alldata.A = sliceseq(A, ppoit.pol.seqlength)

    # Carefull, overwriting r here
    applydiscount!(r, ppoit.rl.gamma)
    meancosts = mean(sum(r, dims = 3))
    ppoit.costvec[episN] = meancosts
    (mod(episN, ppoit.printevery) == 0 || episN == 1) && println("\rEpis. $episN : mean costs $(meancosts) \u1b[K")

    # Set other variables
    ppoit.newepisode[1] = false

    return 1
end

function getbatch(ppodata::PPOEpoptData, e_perc::Number, batchsize::Int, pol::Policy)
    maxind = floor(Int, size(ppodata.X, 2)*e_perc)
    indices_unsorted = randperm(maxind)[1:batchsize] # random indices from 1-maxind (1 would be worst traj)
    indices = ppodata.sort_ind[indices_unsorted] # indices of worst traj
    X = pol.atype(view(ppodata.X, :, indices, :))
    U = pol.atype(view(ppodata.U, :, indices, :))
    meanUold = pol.atype(view(ppodata.meanUold, :, indices, :))
    A = pol.atype(view(ppodata.A, :, indices, :))
    return PPOInput{pol.atype}(X, U, meanUold, A)
end
