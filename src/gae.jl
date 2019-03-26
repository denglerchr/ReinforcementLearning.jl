# functions for the generalised advantage estimation
function gae(rl::PolicyGradientAlgorithm, X, r)
    # Compute the value function
    @assert (rl.nH == size(X, 1))
    nTraj = size(X, 2)
    nsteps = size(X, 3)
    V = Array{Float64}( rl.valfunc(rl.atype(X)) )

    # Compute the advantage
    A = Array{Float64}(undef, size(V))
    delta_V = Array{Float64}(undef, nTraj)
    delta_V1 = Array{Float64}(undef, nTraj)
    fill!(A, 0)
    fill!(delta_V, 0)
    fill!(delta_V1, 0)
    discount = rl.gamma*rl.lambda

    for T = nsteps-1:-1:1
        delta_V1 .= delta_V
        delta_V .= r[1, :, T] .+ rl.gamma.*V[1, :, T+1] .- V[1, :, T]
        A[1, :, T] .= delta_V .+ discount.*delta_V1
    end

    return A
end