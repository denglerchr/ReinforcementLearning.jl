struct Reinforce <: PolicyGradientAlgorithm
    # Hyperparameters
    Ntraj::Int # Number of trajectories
    Nsteps::Number # Max number of discrete time-steps before a trajectory is finished
    Nepisodes::Int # Number of episodes
    gamma::Number # Discount factor

    baseline::Symbol # Baseline used to reduce the variance of the gradient. Chose in {:none, :mean}

    printevery::Int # Print loss every xx episodes
    workerpool::WorkerPool
end
Reinforce(Ntraj::Int, Nsteps::Number, Nepisodes::Int, gamma::Number, baseline::Symbol) = Reinforce(Ntraj, Nsteps, Nepisodes, gamma, baseline, 1,  default_worker_pool())


# This is used to generate data batches using the iterate function
struct ReinforceIterator
    env::Environment
    pol::Policy
    rl::Reinforce
end

# Main function
function minimize!(rl::Reinforce, pol::Policy, env::Environment)
    @assert pol.nX == env.nX
    @assert pol.nU == env.nU

    costvec = [NaN for i = 1:rl.Nepisodes]

    # Define loss and batch generator
    loss(X, U, r, episN, printbool) = logpdfr!(pol, rl.baseline, X, U, r, episN, printbool, costvec)
    rlit = ReinforceIterator(env, pol, rl)

    # Compute gradient using backprop and apply gradient
    Knet.minimize!(loss, rlit, pol.optimizer)

    return costvec
end


## Iterator function, used to generate the data batches
# iterate to create one batch of data
function iterate(rlit::ReinforceIterator)
    # Get trajectories
    print("\rEpis. 1\r")
    trajvec = gettraj(rlit.pol, rlit.env, rlit.rl)

    # Transform the data into big tensors for faster GPU computation
    X, U, r = stacktraj(trajvec, rlit.pol)
    applydiscount!(r, rlit.rl.gamma)
    atype = rlit.pol.atype
    return (atype(X), atype(U), atype(r), 1, true), 1
end

function iterate(rlit::ReinforceIterator, episodeN::Int)
    # Finish after sampling Nepisodes episodes
    episodeN >= rlit.rl.Nepisodes && return nothing

    newepisN = episodeN+1

    # Decide if the loss function call should print something or not
    printbool = ( mod(newepisN, rlit.rl.printevery) == 0)

    # If not finished, get new trajectories etc. (Same as above)
    print("\rEpis. $newepisN\r")
    trajvec = gettraj(rlit.pol, rlit.env, rlit.rl)
    X, U, r = stacktraj(trajvec, rlit.pol)
    applydiscount!(r, rlit.rl.gamma)
    atype = rlit.pol.atype
    return (atype(X), atype(U), atype(r), newepisN, printbool), newepisN
end

length(it::ReinforceIterator) = it.rl.Nepisodes


## The actual gradient generating functions
# log(p)*R without baseline.
function logpdfr!(pol::Policy, baseline::Symbol, X::T, U::T, r::T, episN::Int, printbool::Bool, costvec::AbstractVector) where {T<:Union{AbstractArray, KnetArray}}

    # mean of the policy when observing these states
    resetpol!(pol)
    mU = umean1(pol, X)

    # log-probability of taking the U's that were actually taken
    p = logpdfgaussian(mU, U, pol.std)
    p = sum(p, dims = 3)

    # Total reward
    R = sum(r, dims = 3)
    meanR = mean(R)
    costvec[episN] = meanR
    if baseline == :mean
        R .-= meanR
    elseif baseline != :none
        @error("Unknown baseline $baseline")
    end

    # Print if requested
    printbool && println("Epis. $episN: mean costs: $meanR")

    # Return the mean of the product, no baseline applied
    return mean(R .* p)
end
