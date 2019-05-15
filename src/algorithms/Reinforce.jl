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
struct ReinforceIterator{P<:Policy}
    env::Environment
    pol::P
    rl::Reinforce
    costvec::Vector
end

# Main function
function minimize!(rl::Reinforce, pol::Policy, env::Environment)
    @assert pol.nX == env.nX
    @assert pol.nU == env.nU

    costvec = [NaN for i = 1:rl.Nepisodes]

    # Define loss and batch generator
    loss(X, U, R) = logpdfr!(pol, X, U, R)
    rlit = ReinforceIterator(env, pol, rl, costvec)

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
    X, U, R = reinforceinputdata(trajvec, rlit.pol, rlit.rl.gamma, rlit.rl.baseline, true, 1, rlit.costvec)

    atype = rlit.pol.atype
    return (atype(X), atype(U), atype(R)), 1
end

function iterate(rlit::ReinforceIterator, episodeN::Int)
    # Finish after sampling Nepisodes episodes
    episodeN >= rlit.rl.Nepisodes && return nothing

    # If not finished, get new trajectories etc.
    newepisN = episodeN+1
    printbool = ( mod(newepisN, rlit.rl.printevery) == 0)

    print("\rEpis. $newepisN")
    trajvec = gettraj(rlit.pol, rlit.env, rlit.rl)
    X, U, R = reinforceinputdata(trajvec, rlit.pol, rlit.rl.gamma, rlit.rl.baseline, printbool, newepisN, rlit.costvec)

    atype = rlit.pol.atype
    return (atype(X), atype(U), atype(R)), newepisN
end

length(it::ReinforceIterator) = it.rl.Nepisodes


function reinforceinputdata(trajvec::Vector{Trajectory}, pol::StaticPolicy, gamma::Number, baseline::Symbol, printbool::Bool, episN::Int, costvec::Vector)

    X, U, r = stackXUr(trajvec, pol)
    applydiscount!(r, gamma)

    # Total reward
    R = sum(r, dims = 3)
    meanR = mean(R)
    printbool && println("\rEpis. $episN, mean costs: $meanR")
    costvec[episN] = meanR
    if baseline == :mean
        R .-= meanR
    elseif baseline == :normalize
        R .-= meanR
        R ./= Statistics.std(R)
    elseif baseline != :none
        @error("Unknown baseline $baseline")
    end

    return X, U, R
end

function reinforceinputdata(trajvec::Vector{Trajectory}, pol::RecurrentPolicy, gamma::Number, baseline::Symbol, printbool::Bool, episN::Int, costvec::Vector)
    X, U = stackXU(trajvec, pol, pol.seqlength)
    r = stackr(trajvec, pol)
    applydiscount!(r, gamma)

    # Total reward
    R = sum(r, dims = 3)
    meanR = mean(R)
    printbool && println("\rEpis. $episN, mean costs: $meanR")
    costvec[episN] = meanR
    if baseline == :mean
        R .-= meanR
    elseif baseline == :normalize
        R .-= meanR
        R ./= Statistics.std(R)
    elseif baseline != :none
        @error("Unknown baseline $baseline")
    end

    # Need to replicate R to the corresponding slices
    N = floor(Int, size(r, 3)/pol.seqlength )
    R = repeat(R, inner = (1, N, 1))

    return X, U, R
end


## The actual gradient generating functions
# log(p)*R without baseline.
function logpdfr!(pol::Policy, X::T, U::T, R::T) where {T<:Union{AbstractArray, KnetArray}}

    # mean of the policy when observing these states
    resetpol!(pol)
    mU = umean1(pol, X)

    # log-probability of taking the U's that were actually taken
    p = logpdfgaussian(mU, U, pol.std)
    p = mean(p, dims = 3)

    # Return the mean of the product, no baseline applied
    return mean(R .* p)
end
