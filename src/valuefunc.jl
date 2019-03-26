# This file contains functions related to training a value function.

# Train a value function.
# X and r should be 3d Tensors, with the time in the 3rd dimension.
function valuetrain!(rl::RLAlgorithm, X, r)
    # Get the target for the value function, depending on the setting.
    # Can be Monte-Carlo (:MC) or TD0 (:TD0)
    @assert(rl.nH == size(X, 1))
    vtarget = valuetarget(r, rl, X, Val(rl.valmethod))
    
    # Flatten time dimension
    X = reshape(X, size(X, 1), 1, size(X, 2)*size(X, 3))
    vtarget = reshape(vtarget, size(vtarget, 1), 1, size(vtarget, 2)*size(vtarget, 3))

    # Split datasets into training and testing #TODO
    percenttrain = 0.8
    indices = randperm(size(X, 3))
    trainindices = indices[1:ceil(Int, length(indices)*percenttrain)]
    testindices = indices[ceil(Int, length(indices)*percenttrain)+1:end]
    Xtrain = rl.atype( X[:, :, trainindices] )
    Xtest = rl.atype( X[:, :, testindices] )
    Ytrain = rl.atype( vtarget[:, :, trainindices] )
    Ytest = rl.atype( vtarget[:, :, testindices] )

    # Create minibatch iterator of the data
    batchsize = max( ceil(Int, size(Xtrain, 3)/20), min( size(Xtrain, 3), 512 ) ) #TODO should this be a parameter of the algorithm, or is a standard value ok
    data = Knet.minibatch(Xtrain, Ytrain, batchsize; shuffle = true)

    # Train the value function
    oldtesterror = Inf
    testerror = Inf
    trainerror = Inf
    stopcount = 0 # increases each time error increases, stop after stopcount = 3
    epoch = 0
    loss(x, y) = mean(abs2, rl.valfunc(x)-y)
    while (stopcount < 3 && epoch<200 && trainerror>1e-5 && testerror>1e-5) # Stop after max 200 epochs to avoid overfitting
        # The main training step
        Knet.minimize!(loss, repeat(data, 5))
        epoch += 5

        # Print progress
        trainerror = loss(Xtrain, Ytrain)
        testerror = loss(Xtest, Ytest)
        print("\t\tTraining value function: epoch $epoch: trainerror: $trainerror, testerror: $testerror\u1b[K\r")

        # Evaluate stopping criteria
        testerror>=oldtesterror && (stopcount += 1) # stop if testerror increasing
        oldtesterror = testerror
    end
    return 1
end

@inline function valuetarget(r, rl::RLAlgorithm, X, ::Val{:MC})
    vtarget = copy(r)
    vtarget[:, :, end] .= 0.0
    for t = size(r, 3)-1:-1:1
        vtarget[:, :, t] .+= vtarget[:, :, t+1].*rl.gamma
    end
    return vtarget
end

@inline function valuetarget(r, rl::RLAlgorithm, X, ::Val{:TD0})
    vtarget = similar(r)
    X2 = rl.atype(X[:, :, 2:end])
    vtarget[:, :, 1:end-1] .= r[:, :, 1:end-1] .+ rl.gamma .* Array( rl.valfunc(X2) )
    vtarget[:, :, end] .= 0.0
    return vtarget
end