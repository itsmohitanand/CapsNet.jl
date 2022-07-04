using Flux:onehot
using StatsBase

function expand_dims(X, which_dim::Int)
    s = [size(X)...]
    insert!(s, which_dim, 1)
    return reshape(X, s...)
end

function squared_norm(X, dims)
    sum(abs2, X, dims = dims)
end

function squash(s)
    epsillon = 1e-7
    s_norm = squared_norm(s, 2)
    s_norm./(1 .+s_norm) .* s ./ (s_norm .+ epsillon)
end

function capsmul!(C, w, u)
    for i in axes(w, 1), j in axes(w, 2), k in axes(u, 4)
        @views mul!(C[i,:,:,j, k], w[i,j,:,:], u[i,:, :,k])
    end
    return C[:,:,1,:,:]
end

function create_onehot(y::AbstractArray)
    oh = Array{Float32, 2}(undef, 10, size(y)[1] )
    for i=0:9
        oh[i+1, :] = onehot(i, y)
    end
    return oh
end

function safe_norm(v)
    ϵ = 1e-7
    sq_norm = squared_norm(v, 1)[1, :, :]
    return sqrt.(sq_norm.+ϵ)
end

function loss_function(v, reconstructed_img, y, img)
    prediction  = safe_norm(v)
    
    left_margin = abs2.(max.(0.0, 0.9.-prediction))
    right_margin = abs2.(max.(0.0, prediction .- 0.1))

    l = y.*left_margin .+ 0.5.* (1.0.-y).*right_margin

    margin_loss = mean(sum(l, dims= 1))
    
    reconstruction_loss = mean(abs2, img[:,:,1,:] .- reconstructed_img)

    margin_loss + 0.0005*reconstruction_loss
end