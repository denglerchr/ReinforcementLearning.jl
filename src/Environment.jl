"""
Defines the environment that the agent is acting in.
Some fields have a different meaning, depending on the algorithm in use.

Fields are:
* nU: The lentgh of the action vector/number of discrete actions
* nX: The lentgh of the state vector/number of discrete states
* dynamics: Callable object, defining the environments behavior.
* resetenv!: A function which return an itial state and eventually resets hidden states of the environment.
"""
struct Environment{T}
    nU::Int # Dimension of the action vector/Number of discrete actions
    nX::Int # Dimension of the state/observation vector/Number of discrete states
    dynamics::T # Dynamics function or struct (if non-Markov case use struct), must be callable. Call using x_{t+1} = dynamics(x_t, u_t).

    # Returns/samples the initial state when generating a new trajectory. The call is resetenv!(dynamics).
    # Also use this in the case of a non-Markov environment to reset the hidden state
    # before sampling a new trajectory.
    resetenv!::Function
end