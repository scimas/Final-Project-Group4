function load_saved_model()
    @load "model.bson" model
    model
end

function test_model(model, X_test, y_test; use_gpu=true)
    if use_gpu
        to_gpu = gpu
        CuArrays.allowscalar(false)
    else
        to_gpu = identity
    end
    model = to_gpu(model)
    testmode!(model)
    loss(ŷ, y) = Flux.logitcrossentropy(ŷ, y)
    test_loss = loss(model(to_gpu(X_test)), to_gpu(y_test))
    println("Testing loss: $(test_loss)")
    model = cpu(model)
end
