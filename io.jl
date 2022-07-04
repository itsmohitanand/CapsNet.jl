
include("utils.jl")

function get_train_data(batch_size)
    
    x_train, y_train = MLDatasets.MNIST.traindata(Float32)
    x_train = x_train/255.0

    x_train = Float32.(expand_dims(x_train, 3))
    y_train = Float32.(create_onehot(y_train))

    return DataLoader((x_train, y_train), batchsize = batch_size, shuffle = true)
end

function get_test_data(batch_size)

    x_test, y_test = MLDatasets.MNIST.testdata(Float32)
    x_test = x_test/255.0
    x_test = Float32.(expand_dims(x_test,3))

    y_test = Float32.(create_onehot(y_test))

    return DataLoader((x_test, y_test), batchsize = batch_size, shuffle = true)
end