
using Flux
using Flux: @epochs, onehotbatch, throttle
using Statistics: mean, var, std
using Plots
pyplot()

add_dim(x::Array) = reshape(x, (1,size(x)...));

TR = Float32;
TC = ComplexF32;

M = 16;
constellation_dim = 2;
N = 32*M;
SNR = 20;
SNRlin = 10^(SNR/10) |> TR;

encoder = Chain(Dense(M, 32, Flux.relu), Dense(32, 32, Flux.relu), Dense(32, constellation_dim));
decoder = Chain(Dense(constellation_dim, 32, Flux.relu), Dense(32, 32, Flux.relu), Dense(32, M));

function model(X)
    X_seed = Flux.onehotbatch(1:M,1:M)
    s_seed = encoder(X_seed)
    s_seed = add_dim(s_seed[1,:] + 1im*s_seed[2,:])
    norm_factor = sqrt(mean(abs.(s_seed).^2))
    
    s = encoder(X)
    s = add_dim(s[1,:] + 1im*s[2,:]) / norm_factor
    𝜎 = sqrt(1/SNRlin) |> TR
    
    r = s + 𝜎 * randn(TC, 1, N)
    r = [real(r); imag(r)]
    Y = decoder(r)
    return Y
end    

loss(X) = Flux.logitcrossentropy(model(X), X);

opt = ADAM(0.001);
ps = params(encoder, decoder);

X = Flux.onehotbatch(rand(1:M, N), 1:M)
data = [[X]]

evalcb() = @show(loss(X));
@epochs 2000 Flux.train!(loss, ps, data, opt, cb = throttle(evalcb, 5));

X_seed = Flux.onehotbatch(1:M,1:M)
s_seed = encoder(X_seed)
s_seed_cpx = add_dim(s_seed[1,:] + 1im*s_seed[2,:])
norm_factor = sqrt(mean(abs.(s_seed_cpx).^2))

s = encoder(X)
s = add_dim(s[1,:] + 1im*s[2,:]) / norm_factor
𝜎 = sqrt(1/SNRlin) |> TR

r = s + 𝜎 * randn(TC, 1, N)
r = [real(r); imag(r)]

mean(abs.(s).^2)

scatter(Flux.Tracker.data(s_seed[1,:]),Flux.Tracker.data(s_seed[2,:]), markershape = :hexagon)

scatter(Flux.Tracker.data(r[1,:]),Flux.Tracker.data(r[2,:]), markershape = :hexagon)
