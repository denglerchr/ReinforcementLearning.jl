# Pendulum with cart, 0 state at bottom
@everywhere using PendulumDyn
import PendulumDyn.Pendulum

@everywhere function rl_pend(x, u)
    pend = PendulumDyn.Pendulum(Float32)
    xt1 = PendulumDyn.pendulum_rk4(x, u[1], 0.05f0, pend)
    costs = xt1[1]^2 + xt1[3]^2 # costs for being away from 0 state
    costs += xt1[1]*xt1[2] # costs for cart moving away from 0 state
    return xt1, costs
end

env = Environment(1, 4, rl_pend, x->Float32[rand(Float32)*2-1, 0.0, 0.0, 0.0])
