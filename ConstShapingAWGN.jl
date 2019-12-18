# This example seems to work

using Flux, Zygote
using Flux: @nograd, @epochs, onehotbatch, throttle, params
using Statistics: mean
using Plots

add_dim(x::Array) = reshape(x, (1,size(x)...))

TR = Float32
TC = ComplexF32

M = 64
constellation_dim = 2
N = 4 * M
SNR = 20
SNRlin = 10^(SNR / 10) |> TR;

encoder = Chain(Dense(M, 32, Flux.relu), Dense(32, 32, Flux.relu), Dense(32, constellation_dim))
decoder = Chain(Dense(constellation_dim, 32, Flux.relu), Dense(32, 32, Flux.relu), Dense(32, M))

function model(X)
    X_seed = Flux.onehotbatch(1:M,1:M)
    s_seed = encoder(X_seed)
    s_seed = add_dim(complex.(s_seed[1,:], s_seed[2,:]))
    norm_factor = sqrt(mean(abs.(s_seed).^2))
    s_seed = s_seed / norm_factor

    s = encoder(X)
    s = add_dim(complex.(s[1,:], s[2,:])) / norm_factor
    𝜎 = sqrt(1/SNRlin) |> TR

    r = s + 𝜎 * randn(TC, 1, N)
    r = [real(r); imag(r)]
    Y = decoder(r)
    return Y , s_seed
end

function loss(x)
    Y, s_seed = model(x)
    return Flux.logitcrossentropy(Y, x)
end

@nograd onehotbatch
X = onehotbatch(rand(1:M, N), 1:M)
@show loss(X)

opt = ADAM(0.001)
ps = params(encoder, decoder)
data = [[X]]

evalcb() = @show(loss(X));
@epochs 4000 Flux.train!(loss, ps, data, opt, cb = throttle(evalcb, 5));

Y, s_seed = model(X)

@show mean(abs.(s_seed).^2)

scatter(real(s_seed)[1,:], imag(s_seed)[1,:], aspect_ratio = :equal, markershape = :hexagon)
ylims!((-2,2))
xlims!((-2,2))
