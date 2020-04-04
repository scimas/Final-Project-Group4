module Testing
include("ASL.jl")

using .ASL
using Flux
using Flux.Optimise: Momentum
using Random: randperm
using BSON: @save, @load

X, y = load_train_data()
println("Data loaded")
println("Size X: $(size(X)) | Size y: $(size(y))")
train_inds = randperm(floor(Int, size(y, 2) * 0.7))
valid_inds = [i for i in 1:size(y, 2) if i ∉ train_inds]
X_train, y_train = X[:, :, :, train_inds], y[:, train_inds]
X_valid, y_valid = X[:, :, :, valid_inds], y[:, valid_inds]
train_loader = Flux.Data.DataLoader(X_train, y_train; batchsize=64)

model = ResNet10(1, 28, 26)
println("Model created")
loss(ŷ, y) = Flux.logitcrossentropy(ŷ, y)
optimizer = Momentum(0.01)

train!(model, loss, optimizer, train_loader, X_valid, y_valid; use_gpu=true)

@load "model.bson" model
testmode!(model)

X, y = load_test_data()
println("Testing data loaded")
println("Size X: $(size(X)) | Size y: $(size(y))")
model = gpu(model)
test_loss = loss(model(gpu(X)), gpu(y))
println("Testing loss: $(test_loss)")
end
