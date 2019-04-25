struct PPO <: PolicyGradientAlgorithm
    # Hyperparameters
    Ntraj::Int # Number of trajectories
    Nsteps::Number # Max number of discrete time-steps before a trajectory is finished
    Nepisodes::Int # Number of episodes
    gamma::Number # Discount factor
    epsilon::Number # Clipping factor for PPO

    # Parameters related to the Value function
    nH::Int # Dimension of the input into the advantage function. Should be env.nX for Markov environments.
    valfunc # Advantage function, should contain Knet.Params to train it
    atype::Type
    usegpu::Bool
    optimizer  # The optimizer for the Advantage function, e.g. Knet.Adam()
    valmethod::Symbol # Defines how the value function targets are computed. Can be :MC or :TD0.
    lambda::Number # For the generalised advantage estimation (gae), TD-lambda

    printevery::Int # Print loss every xx episodes
    workerpool::WorkerPool
end


# This is used to return multiple times the same data while iterating
mutable struct PPOinput{T}
    X::T
    H::T # is not used in case of a static policy
    U::T
    meanUold::T
    A::T
end


struct PPOIterator{P<:Policy}
    env::Environment
    pol::P
    rl::PPO
    lossinput::PPOinput
    costvec::Vector{Float64}
    newepisode::Vector{Bool}
end


# The main call, starting the rl optimization
function minimize!(rl::PPO, pol::Policy, env::Environment)
    @assert pol.nX == env.nX
    @assert pol.nU == env.nU
    @assert pol.usegpu == rl.usegpu
    @assert pol.atype == rl.atype

    costvec = [NaN for i = 1:rl.Nepisodes]

    # Define loss and batch generator
    lossfun(ppoinput) = ppoloss!(pol, rl.epsilon, ppoinput)
    ppores = PPOinput(pol.atype(zeros(1, 1, 1)), pol.atype(zeros(1, 1, 1)), pol.atype(zeros(1, 1, 1)), pol.atype(zeros(1, 1, 1)), pol.atype(zeros(1, 1, 1)))
    ppoit = PPOIterator(env, pol, rl, ppores, costvec, [true])

    # Compute gradient using backprop and apply gradient
    optiter = Knet.minimize(lossfun, ppoit, pol.optimizer) # TODO create iterator, which produces the XUr tuple...
    next = iterate(optiter)
    oldloss = Inf
    newepcount = 0 # increases when the loss increases. Sample new trajectories after newepcount=3
    while next !== nothing
        (loss, state) = next
        if (loss>=oldloss)
            newepcount += 1
            newepcount>3 && (ppoit.newepisode[1] = true; newepcount = 0)
            oldloss = Inf
        else
            oldloss = loss
        end
        next = iterate(optiter, state)
    end

    return ppoit.costvec
end

## Functions for a static policy
# First iterate call, producting data that is fed into the forward pass
function iterate(ppoit::PPOIterator{<:P}) where {P<:Policy}
    getnewlossinput!(ppoit, 1)
    return (ppoit.lossinput,), (1, 1)
end

# first state is the episode number, second state is a counter for ppo, reusing old data to produce yet another gradient
function iterate(ppoit::PPOIterator{<:P}, state::Tuple{Int, Int}) where {P<:Policy}

    episN = state[1]
    incount = state[2]
    # check if we should do another gradient with old data
    if (ppoit.newepisode[1] == false && incount < 20)
        print("Epis. $episN , $(incount+1) \u1b[K\r")
        return (ppoit.lossinput,), (episN, incount+1)
    end

    ## Pass on to next episode, update lossinput
    newepisN = episN + 1
    newepisN > ppoit.rl.Nepisodes && return nothing

    getnewlossinput!(ppoit, newepisN)

    return (ppoit.lossinput,), (newepisN, 1)
end


# Sample new trajectories, train value function etc etc
function getnewlossinput!(ppoit::PPOIterator{<:StaticPolicy}, episN::Int)
        # Get trajectories
        print("Epis. $episN , 1 \r")
        trajvec = gettraj(ppoit.pol, ppoit.env, ppoit.rl)

        # Transform the data into big tensors for faster GPU computation
        X, U, r = stacktraj(trajvec, ppoit.pol)
        ppoit.lossinput.X = ppoit.rl.atype(X)
        ppoit.lossinput.U = ppoit.rl.atype(U)

        # Train the value function
        valuetrain!(ppoit.rl, X, r)

        # Compute the "generalised advantage estimation"
        A = gae(ppoit.rl, X, r)
        ppoit.lossinput.A = ppoit.rl.atype(A)

        # Carefull, overwriting r here
        applydiscount!(r, ppoit.rl.gamma)
        meancosts = mean(sum(r, dims = 3))
        ppoit.costvec[episN] = meancosts
        (mod(episN, ppoit.rl.printevery) == 0 || episN == 1) && println("Epis. $episN : mean costs $(meancosts) \u1b[K") # \u1b[K cleans the rest of the line

        # Compute the meanU of the (old) policy
        meanUold = ppoit.pol.umean(ppoit.lossinput.X)
        ppoit.lossinput.meanUold = meanUold

        # Set other variables
        ppoit.newepisode[1] = false

        return 1
end

function ppoloss!(pol::StaticPolicy, epsilon::Number, ppoinput::PPOinput)
    # Compute the quotient of probabilities pol_new(U)/pol_old(U)
    meanUnew = pol.umean(ppoinput.X)
    probquot = pdfgaussianquot(meanUnew, ppoinput.meanUold, ppoinput.U, pol.std, pol.std)

    # Set gradient to zero for quotient too far from 1
    clipbool = (ppoinput.A .> 0) # 1 where probquot should be clipped at 1-epsilon, 0 if clipping at 1+epsilon
    probquotclipped = clipbool.*max.(1.0-epsilon, probquot) .+ (1 .- clipbool) .* min.(1.0+epsilon, probquot)

    # return the mean of the product, which is what is minimized here (usually called "L")
    return mean(probquotclipped.*ppoinput.A)
end


## Functions for a recurrent policy

# Sample new trajectories, train value function etc etc
function getnewlossinput!(ppoit::PPOIterator{<:RecurrentPolicy}, episN::Int)
    # Get trajectories
    print("Epis. $episN , 1 \r")
    trajvec = gettrajh(ppoit.pol, ppoit.env, ppoit.rl)

    # Transform the data into big tensors for faster GPU computation
    X, H, U, r = stacktrajh(trajvec, ppoit.pol)
    ppoit.lossinput.X = ppoit.rl.atype(X)
    ppoit.lossinput.H = ppoit.rl.atype(H)
    ppoit.lossinput.U = ppoit.rl.atype(U)

    # Train the value function
    valuetrain!(ppoit.rl, H, r)

    # Compute the "generalised advantage estimation"
    A = gae(ppoit.rl, H, r)
    ppoit.lossinput.A = ppoit.rl.atype(A)

    # Carefull, overwriting r here
    applydiscount!(r, ppoit.rl.gamma)
    meancosts = mean(sum(r, dims = 3))
    ppoit.costvec[episN] = meancosts
    (mod(episN, ppoit.rl.printevery) == 0 || episN == 1) && println("Epis. $episN : mean costs $(meancosts) \u1b[K")

    # Compute the meanU of the (old) policy
    resetpol!(ppoit.pol)
    meanUold = umean1(ppoit.pol, ppoit.lossinput.X)
    ppoit.lossinput.meanUold = meanUold

    # Set other variables
    ppoit.newepisode[1] = false

    return 1
end


function ppoloss!(pol::RecurrentPolicy, epsilon::T, ppoinput::PPOinput) where {T<:Number}
    # Compute the quotient of probabilities pol_new(U)/pol_old(U)
    resetpol!(pol)
    meanUnew = umean1(pol, ppoinput.X)
    probquot = pdfgaussianquot(meanUnew, ppoinput.meanUold, ppoinput.U, pol.std, pol.std)

    # Set gradient to zero for quotient too far from 1
    # TODO should probably use new H here and recompute A... (problematic as the backprop though V?) as it is the advantage in state H, which changed
    # or get meanUnew from H, but then X->H will not be trained...
    clipbool = (ppoinput.A .> 0) # 1 where probquot should be clipped at 1-epsilon, 0 if clipping at 1+epsilon
    probquotclipped = clipbool.*max.(1-epsilon, probquot) .+ (1 .- clipbool) .* min.(1 + epsilon, probquot)

    # return the mean of the product, which is what is minimized here (usually called "L")
    return mean(probquotclipped.*ppoinput.A)
end
