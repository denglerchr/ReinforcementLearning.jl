mutable struct Qlearning
    Nsteps::Int # Number of steps per episode
    Nepisodes::Int # Number of episodes
    gamma::Number # discount factor
    Qfunc::Matrix # Contains the Q values
end

function minimize!(rl::Qlearning, pol::EpsGreedyPolicy, env::Environment)
    # TODO
    println("This is not yet implemented")
end