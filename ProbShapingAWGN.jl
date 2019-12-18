# This example still fails, not sure where it goes wrong

using Flux, Zygote
using Flux: @nograd, @adjoint, @epochs, onehotbatch, throttle, params
using Distributions
using Distributions: Gumbel
using Statistics: mean
using Plots

add_dim(x::Array) = reshape(x, (1,size(x)...))

function qammod(M)
    r = 1:sqrt(M)
    r = 2 * (r .- mean(r))
    r = [i for i in r, j in r]
    constellation = vcat(complex.(r, r')...)
    norm = sqrt(mean(abs2.(r)))
    constellation / norm
end

p_norm(p, x, fun) = sum(p .* fun(x))

function straight_through_estimator(x)
    M = size(x)[1]
    min_idx = argmin.(eachcol(x))
    Flux.onehotbatch(min_idx, 1:M)
end
# forward function, backward identity
@adjoint straight_through_estimator(x) = straight_through_estimator(x), x -> (x,)

TR = Float32
TC = ComplexF32

M = 64
constellation_dim = 2
N = 32 * M
temperature = 1.
SNR = 15
SNRlin = 10^(SNR/10) |> TR

nHidden = 128

constellation = qammod(M)
g_dist = Gumbel()

encoder = Chain(Dense(1, nHidden, Flux.relu),
                Dense(nHidden, M));
decoder = Chain(Dense(constellation_dim, nHidden, Flux.relu),
                Dense(nHidden, nHidden, Flux.relu),
                Dense(nHidden, M));

function gumbel_sample(M, N)
    rand(g_dist, (M, N))
end

function model(X)
    # sample from discrete distribution
    s_logits = encoder(X)
    g = gumbel_sample(M, N)
    s_bar = Flux.softmax((g .+ s_logits) / temperature)
    s = straight_through_estimator(s_bar)
    p_s = Flux.softmax(s_logits)

    # modulation
    norm_factor = sqrt(p_norm(p_s, constellation, x -> abs2.(x)))
    norm_constellation = constellation / complex.(norm_factor, 0)
    x = add_dim(norm_constellation) * complex.(s, 0)

    # Channel
    𝜎 = sqrt(1 / SNRlin) |> TR
    r = x + 𝜎 * randn(TC, 1, N)
    r = [real(r); imag(r)]

    # decoder
    Y = decoder(r)
    return p_s, s, Y, x, norm_constellation
end

stop_gradient(x) = x
@nograd stop_gradient

function loss(X)
    p_s, s, Y, x, norm_constellation = model(X)
    logit_loss = Flux.logitcrossentropy(Y, stop_gradient(s))
    entropy_x = -p_norm(p_s, p_s, x -> log2.(x))
    logit_loss - entropy_x
end

X = ones(1, 1)

@show loss(X)

ps = params(encoder, decoder);

gs = gradient(ps) do
    loss(X)
end

@show gs[encoder[1].W]

# opt = ADAM(0.001);
# data = [[X]]
# evalcb() = @show(loss(X));
# @epochs 500 Flux.train!(loss, ps, data, opt, cb = throttle(evalcb, 5));
#
# p_s, s, Y, x, norm_constellation = model(X)
#
# scatter(real(norm_constellation), imag(norm_constellation), aspect_ratio = :equal,
#                                                             markershape = :circle,
#                                                             markersize = 500*p_s,
#                                                             markerstrokealpha = 0)
# lim_ = 1.5
# ylims!((-lim_,lim_))
# xlims!((-lim_,lim_))
#
# @show p_s
