# This file contains functions related to training a value function.

# Train a value function.
# X and r should be 3d Tensors, with the time in the 3rd dimension.
function valuetrain!(rl::RLAlgorithm, X, r)
    # Get the target for the value function, depending on the setting.
    # Can be Monte-Carlo (:MC) or TD0 (:TD0)
    @assert(rl.nH == size(X, 1))
    vtarget = valuetarget(r, rl, X, Val(rl.valmethod))

    # Flatten time dimension
    X = reshape(X, size(X, 1), size(X, 2)*size(X, 3))
    vtarget = reshape(vtarget, size(vtarget, 1), size(vtarget, 2)*size(vtarget, 3))

    # Split datasets into training and testing
    percenttrain = 0.8
    indices = randperm(size(X, 2))
    trainindices = indices[1:ceil(Int, length(indices)*percenttrain)]
    testindices = indices[ceil(Int, length(indices)*percenttrain)+1:end]
    Xtrain = X[:, trainindices]
    Ytrain = vtarget[:, trainindices]
    Xtest = rl.atype( X[:, testindices] )
    Ytest = rl.atype( vtarget[:, testindices] )

    # Create minibatch (batchsize between 256 and 2048 Tuples) and iterator for training
    batchsize = max( min( size(Xtrain, 2), 256 ), min(ceil(Int, size(Xtrain, 2)/30), 8192) )
    data = Knet.minibatch(Xtrain, Ytrain, batchsize; shuffle = true, xtype = rl.atype, ytype = rl.atype)

    # Train the value function
    oldtesterror = Inf
    testerror = Inf
    trainerror = Inf
    stopcount = 0 # increases if error increases, stop after stopcount passes threshold
    epoch = 0
    loss(x, y) = mean(abs2, rl.valfunc(x)-y)
    optiter = Knet.minimize(loss, data)
    while (stopcount < 10 && epoch<1000 && trainerror>1e-5 && testerror>1e-5) # Stop after max 2000 epochs to avoid overfitting
        # The main training steps
        sumtrainloss = 0.0 # variables needed to compute the mean trainig error
        trainlosscount = 0 # variables needed to compute the mean trainig error
        for j = 1:5 # do 5 epochs
            next = iterate(optiter)
            while next !== nothing
                (temploss, state) = next
                trainlosscount += 1
                sumtrainloss += temploss
                next = iterate(optiter, state)
            end
        end
        trainerror = sumtrainloss/trainlosscount
        epoch += 5

        # Compute testerror and Print progress
        testerror = loss(Xtest, Ytest)
        print("\t\tTraining value function: epoch $epoch: trainerror: $trainerror, testerror: $testerror\u1b[K\r")

        # Evaluate stopping criteria
        testerror>=oldtesterror ? (stopcount += 2) : (stopcount = max(0, stopcount-1)) # increase "stopcount" if testerror increasing
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
