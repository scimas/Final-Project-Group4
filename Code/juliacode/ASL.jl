__precompile__()
module ASL

using Flux, CuArrays, DataFrames, CSV
using Flux: onehotbatch
using Flux.Optimise: update!
using BSON: @save, @load
using Dates: now

export load_train_data, load_test_data, ResNet10, train!, load_saved_model, test_model

include("preprocess.jl")
include("models.jl")
include("training.jl")
end