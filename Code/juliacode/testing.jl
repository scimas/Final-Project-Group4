include("ASL.jl")

using .ASL
using Flux, CuArrays
using Flux: onehotbatch
using Flux.Optimise: Momentum
using Random: randperm
using MLDataUtils

X, y = load_train_data()
weights = class_weights(y, 0:25)
println("Data loaded")
println("Size X: $(size(X)) | Size y: $(size(y))")
(X_train, y_train), (X_valid, y_valid) = stratifiedobs((X, y); p=0.7, shuffle=false)
y_train = Float32.(onehotbatch(y_train, 0:25))
y_valid = Float32.(onehotbatch(y_valid, 0:25))
train_loader = Flux.Data.DataLoader(X_train, y_train; batchsize=64)

model = ResNet10(1, 28, 26)
println("Model created")
loss(ŷ, y, weights) = Flux.logitcrossentropy(ŷ, y; weight=weights)
optimizer = Momentum(0.01)

train!(model, loss, optimizer, train_loader, X_valid, y_valid; class_weights=weights, use_gpu=true)

model = load_saved_model()
X_test, y_test = load_test_data()
y_test = Float32.(onehotbatch(y_test, 0:25))
test_model(model, X_test, y_test; use_gpu=true)
