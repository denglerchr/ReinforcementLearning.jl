@everywhere import HeavyChain

## Create a callable wrapper
@everywhere struct HCstruct
    chain::HeavyChain.Chain
end


@everywhere function (c::HCstruct)(~, action)
    dt = 0.001
    for i = 1:10
        HeavyChain.simulate!(c.chain, action[1], dt)
    end
    state2 = vcat(c.chain.q[1], c.chain.u[1])
    xend = HeavyChain.getendpoint(c.chain) # position of the bottom of the chain
    reward = xend^2
    return state2, reward
end


# Set the chain into a random position away from the 0 state 
@everywhere function resetenv!(c::HCstruct)
    HeavyChain.resetchain!(c.chain) # Set everything to zero
    c.chain.q[1] = (rand()-0.5) # Set initial position to be in (-0.5, 0.5)
    return Float64[c.chain.q[1], 0.0]
end

## Pack it all into the Environment struct
chaintemp = HeavyChain.createchain()
hc = HCstruct(chaintemp)
env = Environment(1, 2, hc, x->resetenv!(x))