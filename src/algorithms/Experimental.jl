struct Experimental <: PolicyGradientAlgorithm
    # Hyperparameters
    Ntraj::Int # Number of trajectories
    Nsteps::Number # Max number of discrete time-steps before a trajectory is finished
    Nepisodes::Int # Number of episodes
    gamma::Number # Discount factor
    epsilon::Number # Regularising factor

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
mutable struct EXPInput{T}
    X::T
    H::T # is not used in case of a static policy
    U::T
    meanUold::T
    A::T
end


struct EXPIterator{P<:Policy}
    env::Environment
    pol::P
    rl::Experimental
    lossinput::EXPInput
    costvec::Vector{Float64}
    newepisode::Vector{Bool}
end


# The main call, starting the rl optimization
function minimize!(rl::Experimental, pol::Policy, env::Environment)
    @assert pol.nX == env.nX
    @assert pol.nU == env.nU
    @assert pol.usegpu == rl.usegpu
    @assert pol.atype == rl.atype

    costvec = [NaN for i = 1:rl.Nepisodes]

    # Define loss and batch generator
    lossfun(expinput) = exploss!(pol, rl.epsilon, expinput)
    expres = EXPInput(pol.atype(zeros(1, 1, 1)), pol.atype(zeros(1, 1, 1)), pol.atype(zeros(1, 1, 1)), pol.atype(zeros(1, 1, 1)), pol.atype(zeros(1, 1, 1)))
    expit = EXPIterator(env, pol, rl, expres, costvec, [true])

    # Compute gradient using backprop and apply gradient
    optiter = Knet.minimize(lossfun, expit, pol.optimizer)
    next = iterate(optiter)
    oldloss = Inf
    newepcount = 0 # increases when the loss increases. Sample new trajectories after newepcount=3
    while next !== nothing
        (loss, state) = next
        if (loss>=oldloss)
            newepcount += 1
            newepcount>3 && (expit.newepisode[1] = true; newepcount = 0; oldloss=Inf)
        else
            oldloss = loss
        end
        next = iterate(optiter, state)
    end

    return expit.costvec
end


function iterate(expit::EXPIterator{<:P}) where {P<:Policy}
    getnewlossinput!(expit, 1)
    return (expit.lossinput,), (1, 1)
end


function iterate(expit::EXPIterator{<:P}, state::Tuple{Int, Int}) where {P<:Policy}

    episN = state[1]
    incount = state[2]
    # check if we should do another gradient with old data
    if (expit.newepisode[1] == false && incount < 20)
        print("Epis. $episN , $(incount+1) \u1b[K\r")
        return (expit.lossinput,), (episN, incount+1)
    end

    ## Pass on to next episode, update lossinput
    newepisN = episN + 1
    newepisN > expit.rl.Nepisodes && return nothing

    getnewlossinput!(expit, newepisN)

    return (expit.lossinput,), (newepisN, 1)
end


# Scale values in M to contain values between low and up
function scale!(M, low::Number, up::Number)
    @assert up>low
    u = maximum(M)
    l = minimum(M)
    d_old = u-l
    d_new = up-low
    M .= (M .- l) .* (d_new/d_old) .+ low
    return M
end


function getnewlossinput!(expit::EXPIterator{<:StaticPolicy}, episN::Int)
        # Get trajectories
        print("Epis. $episN , 1 \r")
        trajvec = gettraj(expit.pol, expit.env, expit.rl)

        # Transform the data into big tensors for faster GPU computation
        X, U, r = stacktraj(trajvec, expit.pol)
        expit.lossinput.X = expit.rl.atype(X)
        expit.lossinput.U = expit.rl.atype(U)

        # Train the value function
        valuetrain!(expit.rl, X, r)

        # Compute the "generalised advantage estimation"
        A = gae(expit.rl, X, r)
        scale!(A, -1, 1) # Normalize A to have entries between -1 and 1
        A .-= mean(A) # Normalize to have mean zero
        expit.lossinput.A = expit.rl.atype(A)

        # Carefull, overwriting r here
        applydiscount!(r, expit.rl.gamma)
        meancosts = mean(sum(r, dims = 3))
        expit.costvec[episN] = meancosts
        (mod(episN, expit.rl.printevery) == 0 || episN == 1) && println("Epis. $episN : mean costs $(meancosts) \u1b[K") # \u1b[K cleans the rest of the line

        # Compute the meanU of the (old) policy
        meanUold = expit.pol.umean(expit.lossinput.X)
        expit.lossinput.meanUold = meanUold

        # Set other variables
        expit.newepisode[1] = false

        return 1
end


function getnewlossinput!(expit::EXPIterator{<:RecurrentPolicy}, episN::Int)
    # Get trajectories
    print("Epis. $episN , 1 \r")
    trajvec = gettrajh(expit.pol, expit.env, expit.rl)

    # Transform the data into big tensors for faster GPU computation
    X, H, U, r = stacktrajh(trajvec, expit.pol)
    expit.lossinput.X = expit.rl.atype(X)
    expit.lossinput.H = expit.rl.atype(H)
    expit.lossinput.U = expit.rl.atype(U)

    # Train the value function
    valuetrain!(expit.rl, H, r)

    # Compute the "generalised advantage estimation"
    A = gae(expit.rl, H, r)
    expit.lossinput.A = expit.rl.atype(A)

    # Carefull, overwriting r here
    applydiscount!(r, expit.rl.gamma)
    meancosts = mean(sum(r, dims = 3))
    expit.costvec[episN] = meancosts
    (mod(episN, expit.rl.printevery) == 0 || episN == 1) && println("Epis. $episN : mean costs $(meancosts) \u1b[K")

    # Compute the meanU of the (old) policy
    resetpol!(expit.pol)
    meanUold = umean1(expit.pol, expit.lossinput.X)
    expit.lossinput.meanUold = meanUold

    # Set other variables
    expit.newepisode[1] = false

    return 1
end


function exploss!(pol::Policy, epsilon::T, expinput::EXPInput) where {T<:Number}
    # Compute the quotient of probabilities pol_new(U)/pol_old(U)
    resetpol!(pol)
    meanUnew = umean1(pol, expinput.X)
    logdiff = logpdfgaussian(meanUnew, expinput.U, pol.std) .- logpdfgaussian(expinput.meanUold, expinput.U, pol.std)
    probquot = exp.(min.(logdiff, log(T(1e6))))

    a = mean(probquot.*expinput.A)
    b = epsilon*mean(abs2, logdiff)
    println("a+b = $a + $b")
    return a+b
end
