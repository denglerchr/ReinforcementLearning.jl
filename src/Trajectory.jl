# Contains variables created when sampling a trajectory
struct Trajectory
    X::AbstractMatrix # Observed state in each timestep
    U::AbstractMatrix # Control input in each timestep
    r::AbstractVector # Reward in each timestep
end


# This is sampled sometimes when using Recurrent Policies, depending on the algorithm
struct TrajectoryH
    X::AbstractMatrix # Observed state for each timestep
    H::AbstractMatrix # Hidden state for each timestep
    U::AbstractMatrix # Control input for each timestep
    r::AbstractVector # Reward for each timestep
end

# The main call, when producing multiple trajectories (use pmap for parallelization)
function gettraj(pol::Policy, env::Environment, rl::PolicyGradientAlgorithm)
    if pol.usegpu
        poltemp = cpupol(pol)
        g = x -> sampletraj(poltemp, env, rl)
    else
        g = x -> sampletraj(pol, env, rl)
    end

    bs = ceil(Int, rl.Ntraj/max(length(rl.workerpool), 1))
    return pmap(g, rl.workerpool, 1:rl.Ntraj; batch_size = bs)
end


function gettrajh(pol::RecurrentPolicy, env::Environment, rl::PolicyGradientAlgorithm)
    if pol.usegpu
        poltemp = cpupol(pol)
        g = x -> sampletrajh(poltemp, env, rl)
    else
        g = x -> sampletrajh(pol, env, rl)
    end

    bs = ceil(Int, rl.Ntraj/max(length(rl.workerpool), 1))
    return pmap(g, rl.workerpool, 1:rl.Ntraj; batch_size = bs)
end


function sampletraj(pol::Policy, env::Environment, rl::PolicyGradientAlgorithm)
    # Preallocate
    cputype = eltype(pol.atype)
    X = Array{cputype}(undef, pol.nX, rl.Nsteps)
    U = Array{cputype}(undef, pol.nU, rl.Nsteps)
    r = Array{cputype}(undef, rl.Nsteps)
    xt = Array{cputype}(undef, pol.nX)

    xt .= env.resetenv!(env.dynamics)
    resetpol!(pol)

    # Sample the trajectory
    for t = 1:rl.Nsteps
        X[:, t] .= xt
        ut = pol(xt)
        U[:, t] .= ut
        simres = env.dynamics(xt, ut)
        xt .= simres[1]
        r[t] = simres[2]
    end

    # Return a Trajectory struct
    return Trajectory(X, U, r)
end


# Sample a trajectory, also saving the hidden states
function sampletrajh(pol::RecurrentPolicy, env::Environment, rl::PolicyGradientAlgorithm)
    # Prallocate
    cputype = eltype(pol.atype)
    X = Array{cputype}(undef, pol.nX, rl.Nsteps)
    H = Array{cputype}(undef, pol.nH, rl.Nsteps)
    U = Array{cputype}(undef, pol.nU, rl.Nsteps)
    r = Array{cputype}(undef, rl.Nsteps)

    xt = env.resetenv!(env.dynamics)
    resetpol!(pol)

    # Sample the trajectory
    for t = 1:rl.Nsteps
        X[:, t] .= xt
        ut, ht = samplepolh(pol, xt)
        U[:, t] .= ut
        H[:, t] .= ht
        xt, rt = env.dynamics(xt, ut)
        r[t] = rt
    end

    # Return a Trajectory struct
    return TrajectoryH(X, H, U, r)
end


# Bring trajectories of same length into a 3d tensor shape, appropriate for calling neural networks
function stacktraj(trajvec::Vector{Trajectory}, pol::Policy)
    cputype = eltype(pol.atype)
    X = Array{cputype}(undef, pol.nX, length(trajvec), size(trajvec[1].X, 2))
    U = Array{cputype}(undef, pol.nU, length(trajvec), size(trajvec[1].U, 2))
    r = Array{cputype}(undef, 1, length(trajvec), length(trajvec[1].r))
    for (i, traj) in enumerate(trajvec)
        X[:, i, :] .= traj.X
        U[:, i, :] .= traj.U
        r[1, i, :] .= traj.r
    end
    return X, U, r
end


function stacktrajh(trajvec::Vector{TrajectoryH}, pol::Policy)
    cputype = eltype(pol.atype)
    X = Array{cputype}(undef, pol.nX, length(trajvec), size(trajvec[1].X, 2))
    H = Array{cputype}(undef, pol.nH, length(trajvec), size(trajvec[1].H, 2))
    U = Array{cputype}(undef, pol.nU, length(trajvec), size(trajvec[1].U, 2))
    r = Array{cputype}(undef, 1, length(trajvec), length(trajvec[1].r))
    for (i, traj) in enumerate(trajvec)
        X[:, i, :] .= traj.X
        H[:, i, :] .= traj.H
        U[:, i, :] .= traj.U
        r[1, i, :] .= traj.r
    end
    return X, H, U, r
end

function stacktrajh(trajvec::Vector{TrajectoryH}, pol::Policy, seqlength::Int)
    cputype = eltype(pol.atype)
    nseq = sum( floor(Int, size(traj.X, 2)/seqlength ) for traj in trajvec )
    X = Array{cputype}(undef, pol.nX, nseq, seqlength)
    H = Array{cputype}(undef, pol.nH, nseq, seqlength)
    U = Array{cputype}(undef, pol.nU, nseq, seqlength)
    r = Array{cputype}(undef, 1, nseq, seqlength)
    i = 1 # Index of the cutted sequences
    for traj in trajvec
        startindex = 1
        while (startindex+seqlength-1) <= size(traj.X, 2)
            X[:, i, :] .= traj.X[:, startindex:startindex+seqlength-1]
            H[:, i, :] .= traj.H[:, startindex:startindex+seqlength-1]
            U[:, i, :] .= traj.U[:, startindex:startindex+seqlength-1]
            r[1, i, :] .= traj.r[startindex:startindex+seqlength-1]
            i += 1
            startindex += seqlength
        end
    end
    @assert(i==nseq)
    return X, H, U, r
end


# Apply the discount factor to the 3d tensor "r" where t is in the 3rd dimension
function applydiscount!(r, gamma)
    gammatemp = gamma
    for t = 2:size(r, 3)
        r[:, :, t] .*= gammatemp
        gammatemp *= gamma
    end
    return r
end
