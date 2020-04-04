module Test
include("Preprocess.jl")
include("Models.jl")
include("Training.jl")
using .Preprocess, .Models, .Training
using Flux: logitcrossentropy
using Flux.Optimise: Momentum
using Random: randperm

X, y = load_data()
println("Data loaded")
println("Size X: $(size(X)) | Size y: $(size(y))")
train_inds = randperm(floor(Int, size(y, 2) * 0.7))
valid_inds = [i for i in 1:size(y, 2) if i ∉ train_inds]
X_train, y_train = X[:, :, :, train_inds], y[:, train_inds]
X_valid, y_valid = X[:, :, :, valid_inds], y[:, valid_inds]

model = ResNet10(1, 28, 26)
println("Model created")
loss(ŷ, y) = logitcrossentropy(ŷ, y)
optimizer = Momentum(0.001)

train!(model, loss, optimizer, X_train, y_train, X_valid, y_valid; use_gpu=true)
end