function load_train_data()
    df = DataFrame(CSV.File("data/sign_mnist_train.csv"; type=UInt8))
    y = df.label
    images = []
    norm = Normalizer(0.5, 0.5)
    for i in 1:size(df, 1)
        im = norm(Vector{Float32}(df[i,2:end])/255)
        push!(images, reshape(im, 28, 28, 1))
    end
    reduce((x, y) -> cat(x, y; dims=4), images), y
end

function load_test_data()
    df = DataFrame(CSV.File("data/sign_mnist_test.csv"; type=UInt8))
    y = df.label
    images = []
    norm = Normalizer(0.5, 0.5)
    for i in 1:size(df, 1)
        im = norm(Vector{Float32}(df[i,2:end])/255)
        push!(images, reshape(im, 28, 28, 1))
    end
    reduce((x, y) -> cat(x, y; dims=4), images), y
end

struct Normalizer
    μ
    σ
end

function (norm::Normalizer)(im)
    @. (im - norm.μ) / norm.σ
end

function class_weights(y::AbstractVector, n_classes=length(unique(y)))
    classes = sort!(unique(y))
    counts = map(class -> count(x -> x == class, y), classes)
    Float32.(length(y) / n_classes ./ counts)
end
