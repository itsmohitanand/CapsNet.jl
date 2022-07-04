using Flux, MLDatasets
using CairoMakie
using Flux.Data: DataLoader
using Flux: @functor, params
using Parameters: @with_kw
using LinearAlgebra: mul!, norm
using Flux:onehot

include("utils.jl")
include("io.jl")


# Defining the arguments of the paper 

@with_kw mutable struct Args
    ϵ = 1e-7
    m_plus = 0.9
    m_minus = 0.1
    λ = 0.5
    α = 0.0005
    epochs = 50
    r = 3               #routing iteration 
    out_channel1 = 256
    n_primary_capsule = 32
    n_digit_capsule = 10
    primary_capsule_vector = 8
    digit_capsule_vector = 16
    batch_size = 2 
end


## Define Primary Capsule

struct PrimaryCapsule
    conv1
    conv2
end

@functor PrimaryCapsule

PrimaryCapsule(in_channel1::Int, out_channel1::Int, n_capsule::Int, n_vector::Int) = PrimaryCapsule(
    Conv((9,9), in_channel1=>out_channel1, relu ; stride = (1,1)), # 20x20x256x100
    Conv((9,9), out_channel1=>n_capsule*n_vector; stride = (2,2)), # 6x6x256x100 #  no activation 
)

function (primary_capsule::PrimaryCapsule)(x)
    x = primary_capsule.conv1(x)
    x = primary_capsule.conv2(x)
    return x
end


#### Defining Digit Capsule

struct DigitCapsule
    W :: AbstractArray
end

@functor DigitCapsule

DigitCapsule(n_incaps::Int, incaps_n_vector::Int, n_outcaps::Int, outcaps_n_vector::Int) = DigitCapsule(
    randn(n_incaps, n_outcaps, outcaps_n_vector, incaps_n_vector)
)



function (digit_caps::DigitCapsule)(x::AbstractArray)
    size_x = size(x)
    u = reshape(x, (size_x[1]*size_x[2]*32,8,1, :))
    u_hat = ones(1152, 16, 1, 10, 100)

    u_hat = capsmul!(u_hat, digit_caps.W, u)
    
    return u_hat
end   


function dynamic_routing(u_hat::AbstractArray, iteration =3)
    b = zeros(1152, 1, 10 ,100)
    v = zeros(1, 16, 10, 100)
    for _=1:iteration
        c = softmax(b, dims = 3)
        s = sum(c.*u_hat, dims = 1 )
        v = squash(s)
        a = sum(v.*u_hat, dims = 2 )
        b = b .+ a
    end
    return v[1,:,:,:]
end


struct Reconstruction
    dense1
    dense2
    dense3
end

@functor Reconstruction

Reconstruction() = Reconstruction(
    Dense(160, 512, relu),
    Dense(512, 1024, relu),
    Dense(1024, 784, relu),

)

function (reconstruction::Reconstruction)(v::AbstractArray, y::AbstractArray)
    y = expand_dims(y, 1)
    v_masked = v.*y
    v_ = reshape(v_masked, (size(v_masked)[1]*size(v_masked)[2], size(v_masked)[3]))
    reconstructed_image = reconstruction.dense1(v_)
    reconstructed_image = reconstruction.dense2(reconstructed_image)
    reconstructed_image = reconstruction.dense3(reconstructed_image)
    
    return reshape(reconstructed_image, (28,28, :))
end


args = Args()
    
train_loader = get_train_data(args.batch_size)

primary_caps = PrimaryCapsule(1, 256, args.n_primary_capsule, args.primary_capsule_vector)
digit_caps = DigitCapsule(1152, args.primary_capsule_vector, args.n_digit_capsule, args.digit_capsule_vector)
reconstruction = Reconstruction()

opt = ADAM()

ps = Flux.params(primary_caps, digit_caps, reconstruction)

train_steps = 0

for epoch=1:2 #args.epochs
    for (x, y) in train_loader
        loss, grad = Flux.withgradient(ps) do
        
            out_1 = primary_caps(x)
            u_hat = digit_caps(out_1)
            v = dynamic_routing(u_hat)
            re_image = reconstruction(v, y)
            loss_function(v, re_image, y, x)
        end
        print("Training step $train_steps")
        Flux.Optimise.update!(opt, ps, grad)
        train_steps+=1
        print(train_steps)
    end
    print("$epochs")
end
   