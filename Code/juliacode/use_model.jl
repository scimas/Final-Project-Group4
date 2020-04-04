function load_saved_model()
    @load "model.bson" model
    model
end

function test_model(model; use_gpu=true)
    if use_gpu
        to_gpu = gpu
        CuArrays.allowscalar(false)
    else
        to_gpu = identity
    end
    model = to_gpu(model)
    testmode!(model)
    X, y = load_test_data()
    println("Testing data loaded")
    println("Size X: $(size(X)) | Size y: $(size(y))")
    model = gpu(model)
    test_loss = loss(model(to_gpu(X)), to_gpu(y))
    println("Testing loss: $(test_loss)")
    model = cpu(model)
end
