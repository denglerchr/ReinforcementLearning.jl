# This file contains the different policy types that can be used.
# For policy gradient algorithms, use one of the following:
#   StaticPolicy
#   RecurrentPolicy
# For Qlearning, use EpsGreedyPolicy
abstract type Policy end


## For policy gradient algorithms
"""
A policy that is using some kind of recurrence (must be implemented as internal state of the pmean struct).
The parameters to be trained should be of type Knet.Param .
If the gpu should be used to train the parameters of "umean", a function to produce a callable cpu version should be provided. In this case the cpu version is only called with vector inputs \\in R^nX and return (u, h), with u \\in R^nU and h \\in R^nH . The gpu version is called with a 3d tensor input X \\in R^{nX, nTraj, nT} returning U \\in R^{nU, nTraj, nT} and H \\in R^{nH, nTraj, nT}. nT is the number of timesteps, so the time is in the 3rd dimensions, the batchsize in 2nd dimension.
In the case of a cpu only policy, "umean" must accept both the vector and 3d tensor inputs.
"resetpolicy!" which resets eventual hidden states is called on the cpu version.
"""
struct RecurrentPolicy{T} <: Policy
    nX::Int # Number of inputs
    nH::Int # Number of hidden states (only required for some algorithms)
    nU::Int # Number of outputs
    std::Number # The standard deviation of the Gaussian policy

    umean::T # Callable object, returns the mean of the gaussian policy. Is called as umean(x).
    atype::Type # Type of the Arrays used in umean (input will be cast to that same type). E.g. Array{Float32}
    usegpu::Bool # Should be true if the umean is trained on a GPU, false if only CPU.
    converttocpu # if the gpu is being used for training, this function must produce a version of umean that can be evaluate on the cpu
    optimizer  # The optimizer, e.g. Knet.Adam()
    seqlength::Int # Sequence length, used for backprop through time

    # This function is called as resetpolicy!(umean) on the "cpu policy" to
    # reset the hidden state of the policy (e.g. set hidden state to zero)
    resetpolicy!::Function
end


"""
A static policy, which means that it samples u_t each time from the same distribution when receiving the same input x_t.
Should not contain a hidden state.
"""
struct StaticPolicy{T} <: Policy
    nX::Int # Number of inputs
    nU::Int # Number of outputs
    std::Number # The standard deviation of the Gaussian policy

    umean::T # Callable object, returns the mean of the gaussian policy. Is called as umean(x).
    atype::Type
    usegpu::Bool # Should be true if the umean is evaluated on a GPU
    converttocpu
    optimizer  # The optimizer, e.g. Knet.Adam()
end


## For Qlearning
struct EpsGreedyPolicy
    #TODO
end


# Function to convert a policy with memory on the gpu to a temporary cpu policy
function cpupol(pol::RecurrentPolicy)
    umean = pol.converttocpu(pol.umean)
    atype = Array{eltype(pol.atype)} # TODO is this actually used?
    return RecurrentPolicy(pol.nX, pol.nH, pol.nU, pol.std, umean, atype, false, identity, nothing, pol.seqlength, pol.resetpolicy!)
end


function cpupol(pol::StaticPolicy)
    umean = pol.converttocpu(pol.umean)
    atype = Array{eltype(pol.atype)} # TODO is this actually used?
    return StaticPolicy(pol.nX, pol.nU, pol.std, umean, atype, false, identity, nothing)
end


# Evaluate a policy
function (pol::StaticPolicy)(x::AbstractVector) 
    cputype = eltype(pol.atype)
    u = umean1(pol, x) .+ cputype(pol.std)*randn(cputype, pol.nU)
    return u
end

function (pol::RecurrentPolicy)(x::AbstractVector)
    cputype = eltype(pol.atype)
    umean, h = pol.umean(x)
    u = umean .+ cputype(pol.std)*randn(cputype, pol.nU)
    return u
end

function samplepolh(pol::RecurrentPolicy, x)
    cputype = eltype(pol.atype)
    umean, h = pol.umean(x)
    u = umean .+ cputype(pol.std)*randn(cputype, pol.nU)
    return u, h
end

# Convenience functions used to write general code for both recurrent and static neural networks
resetpol!(pol::StaticPolicy) = nothing
resetpol!(pol::RecurrentPolicy) = pol.resetpolicy!(pol.umean)
umean1(pol::StaticPolicy, X) = pol.umean(X)
umean1(pol::RecurrentPolicy, X) = pol.umean(X)[1]
