function train!(model, loss, optimizer, train_loader, X_valid, y_valid; use_gpu=true, max_epochs=100, patience=3)
    if use_gpu
        to_gpu = gpu
        CuArrays.allowscalar(false)
    else
        to_gpu = identity
    end
    model = to_gpu(model)
    min_val_loss = 1f10
    improve_epoch = 1
    println("Starting training loop.")
    for epoch in 1:max_epochs
        tick = now()
        trainmode!(model)
        weights = Flux.params(model)
        train_loss = 0f0
        for (x, y) in train_loader
            grads = gradient(weights) do
                minibatch_loss = loss(model(to_gpu(x)), to_gpu(y))
                train_loss += cpu(minibatch_loss) * size(y, 2)
                minibatch_loss
            end
            update!(optimizer, weights, grads)
        end
        testmode!(model)
        valid_loss = cpu(loss(model(to_gpu(X_valid)), to_gpu(y_valid)))
        tock = now()
        println("Epoch: $(epoch) | Training loss: $(train_loss / size(y_train, 2)) | Validation loss: $(valid_loss) | Time: $(tock - tick)")
        if valid_loss < min_val_loss
            min_val_loss = valid_loss
            improve_epoch = epoch
            model = cpu(model)
            @save "model.bson" model
            model = to_gpu(model)
        elseif epoch - improve_epoch >= patience
            println("Validation loss did not decrease for $(patience) epochs. Breaking from training loop.")
            break
        end
    end
    println("Training complete.")
end
