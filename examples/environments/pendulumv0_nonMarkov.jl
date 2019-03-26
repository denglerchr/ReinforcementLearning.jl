# Non Makrov version of the pendulum problem. Only the angle can be seen, not the angluar velocity.
@everywhere struct Pendulum
    state::AbstractVector
end

@everywhere function (pend::Pendulum)(x, u)
    # the state vector contains the angle and angular velocity
    # 0 angle is at the top
    anglenormalize(in) = mod(in+pi, 2*pi)-pi # angle restricted to -pi, pi
    maxspeed = 8.0
    maxtorque = 2.0

    dt = 0.05
    g = 10.0
    m = 1.0
    l = 1.0

    u = clamp(u[1], -maxtorque, maxtorque)
    costs = anglenormalize(pend.state[1])^2 + 0.1*pend.state[2]^2 + 0.001*(u^2)

    pend.state[2] += (-3*g/(2*l) * sin(pend.state[1] + pi) + 3.0 / (m*l^2)*u) * dt
    pend.state[1] += pend.state[2]*dt
    pend.state[2] = clamp(pend.state[2], -maxspeed, maxspeed)

    return pend.state[1:1], costs
end

pend = Pendulum(zeros(2))

@everywhere resetenv!(pend::Pendulum) = (pend.state[1]=pi+randn()*0.001; pend.state[2] = 0.0; pend.state[1:1])

env = Environment(1, 1, pend, x->resetenv!(x))