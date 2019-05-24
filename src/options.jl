"""
Options
Contains settings:
printevery: A positive integer, how often (in terms of episodes) the mean costs should be printed
saveevery: An integer, how often (in terms of episodes) the policy should be saved to a file. Set <1 if you do not want to save it regularly.
filename: A string, the name of a jld2 file where the policy will be saved every "saveevery" episodes
workerpool: Which workers will be used for the simualations.
"""
struct Options
    printevery::Int
    saveevery::Int
    filename::String
    workerpool::WorkerPool

    function Options(printevery, saveevery, filename, workerpool)
        @assert printevery > 0
        @assert filename[end-4:end] == ".jld2"
        return new(printevery, saveevery, filename, workerpool)
    end
end


function Options()
    return Options(1, 0, "temp.jld2", default_worker_pool())
end


"""
Saves policy and costs if "episN" is a multiple of "savevery".
Return 1 something was written to a file, 0 else.
"""
function Options_savepol(options::Options, pol::Policy, costvec::AbstractVector, episN::Int)
    # Dont save if saveevery is set <1
    options.saveevery<1 && return 0

    # Save if episN is a multiple of saveevery
    if mod(episN, options.saveevery) == 0
        Knet.jldopen(options.filename, "a+") do file
            file[string(episN)*"/cpupol"] = cpupol(pol)
            file[string(episN)*"/cost"] = costvec[episN]
        end
        return 1
    else
        return 0
    end
end
