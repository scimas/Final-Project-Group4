"""
    ResidualBlock

Stores the structure of a residual block (as described in the resnet paper) alongwith its shortcut connection.

It provides the same functionality as built-in Flux layers using the `@functor` macro.
"""
struct ResidualBlock
    layers::Chain
    shortcut
end

Flux.@functor ResidualBlock

function (block::ResidualBlock)(x)
    out1 = block.shortcut(x)
    out2 = block.layers(x)
    relu.(out1 + out2)
end

function ResidualBlock(in_channels, out_channels, downsample::Bool)::ResidualBlock
    if downsample
        conv1 = Conv((3, 3), in_channels => out_channels, relu, stride=(2, 2), pad=(1, 1))
        shortcut = Chain(
            Conv((1, 1), in_channels => out_channels, stride=(2, 2)),
            BatchNorm(out_channels)
        )
    else
        conv1 = Conv((3, 3), in_channels => out_channels, relu, pad=(1, 1))
        if in_channels != out_channels
            shortcut = Chain(
                Conv((1, 1), in_channels => out_channels),
                BatchNorm(out_channels)
            )
        else
            shortcut = identity
        end
    end
    conv2 = Conv((3, 3), out_channels => out_channels, relu, pad=(1, 1))
    norm1 = BatchNorm(out_channels, relu)
    norm2 = BatchNorm(out_channels)
    ResidualBlock(Chain(conv1, norm1, conv2, norm2), shortcut)
end

"""
    ResNet

Stores all of the layers of a ResNet like model.

It provides the same functionality as a model built using Flux layers through the us of the `@functor` macro.
"""
struct ResNet
    layers::Chain
end

Flux.@functor ResNet

function (model::ResNet)(x)
    model.layers(x)
end

"""
    ResNet10(in_channels::Integer, inputs::Integer, outputs::Integer)::ResNet

Creates a 10 layer `ResNet` model. The input is assumed to be square shaped. The number of channels, input size and the output size can be specified using the corresponding parameters.

Note that the output of the model is not normalized and thus cannot be treated as probabilities. You would need to normalize it (using, for example, softmax) for such a use case.
"""
function ResNet10(in_channels::Integer, inputs::Integer, outputs::Integer)::ResNet
    layers = []
    push!(layers, Conv((7, 7), in_channels => 64, relu, stride=(2, 2), pad=(3, 3)))
    inputs = ceil(Int, inputs/2)
    push!(layers, MaxPool((3, 3); stride=(2, 2), pad=(1, 1)))
    inputs = ceil(Int, inputs/2)
    push!(layers, ResidualBlock(64, 64, false))
    push!(layers, ResidualBlock(64, 128, true))
    inputs = ceil(Int, inputs/2)
    push!(layers, ResidualBlock(128, 256, true))
    inputs = ceil(Int, inputs/2)
    push!(layers, ResidualBlock(256, 512, true))
    inputs = ceil(Int, inputs/2)
    push!(layers, x -> reshape(x, :, size(x, 4))) # flatten
    push!(layers, Dense(inputs*inputs*512, outputs))
    ResNet(Chain(layers...))
end
