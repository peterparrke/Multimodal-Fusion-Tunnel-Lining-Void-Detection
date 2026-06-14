function h = hammer_deconvolution(f, y, fs, alpha, fband)

    N = min(length(f), length(y));
    f = double(f(1:N));
    y = double(y(1:N));

    win = hann(N);
    f = f .* win;
    y = y .* win;

    Nfft = 2^nextpow2(N);
    F = fft(f, Nfft);
    Y = fft(y, Nfft);

    eps_reg = alpha * mean(abs(F).^2);
    H = (Y .* conj(F)) ./ (abs(F).^2 + eps_reg);

    freqs = (0:Nfft-1) * fs / Nfft;
    mask = (freqs >= fband(1)) & (freqs <= fband(2));
    H(~mask) = 0;

    h_full = real(ifft(H, Nfft));
    h = h_full(1:N);
end
