# Pendulum WITHOUT a cart, same dynamics as in the openAI gym

@everywhere function pendulumv0(x, u)
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
    costs = anglenormalize(x[1])^2 + 0.1*x[2]^2 + 0.001*(u^2)

    out = similar(x)
    out[2] = x[2] + (-3*g/(2*l) * sin(x[1] + pi) + 3.0 / (m*l^2)*u) * dt
    out[1] = x[1] + out[2]*dt
    out[2] = clamp(out[2], -maxspeed, maxspeed)

    return out, costs
end

env = Environment(1, 2, pendulumv0, x->[-pi+(rand()-0.5)*pi/3, 0.0])