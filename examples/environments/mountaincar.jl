@everywhere function movecar(x::AbstractVector, u::Number)
    out = similar(x, 2)

    # New velocity
    u = clamp(u, -1, 1)
    out[2] = x[2] + 0.001*u - 0.0025*cos(3*x[1])

    # Correct bounds
    out[2] = clamp(out[2], -0.07, 0.07)

    # Update position
    out[1] = x[1] + out[2]

    #check if the car reached a boundary
    if (out[1]<=-1.2)
        out[1] = -1.2;
        out[2] = 0.0;
    elseif (out[1]>=0.5)
        out[1] = 0.5
        out[2] = 0.07
        return out, 0.0;
    end
    #return out, 1.0
    cost = 0.01-0.0025/3.0*( sin(3.0*out[1])-sin(3.0*0.5) )
    return out, cost
end

@everywhere movecar(x::AbstractVector, u::AbstractVector) = movecar(x, u[1])

env = Environment(1, 2, movecar, x->[-pi/6+randn()*0.001, 0.0])