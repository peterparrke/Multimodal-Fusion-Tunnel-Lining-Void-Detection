clc; clear; close all;

%% ============================================================
%  参数设置
% ============================================================
inputDir  = 'C:\GUO\jinqiang\data\sdcq_2';
outputDir = 'C:\GUO\jinqiang\data_viasual\time_fre_deconv';
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

fs    = 1e6;              % 采样率 (Hz)
Nfft  = 131072;           % FFT 点数
alpha = 1e-2;             % 去卷积正则化
fband = [2e3 30e3];       % 有效频带 (Hz)

% ===== 网格信息 =====
nRow = 29;                % 行数
nCol = 64;                % 列数
targetCol = 41;           % 只提取第41列

%% ============================================================
%  1. 读取所有 txt 文件
% ============================================================
files = dir(fullfile(inputDir, 'sdcq*#*.txt'));
nFiles = numel(files);
fprintf('共找到 %d 个 txt 文件。\n', nFiles);

%% ============================================================
%  2. 按测点前缀分组
%     例如 sdcq2963#1 ~ sdcq2963#4 属于同一个测点前缀 sdcq2963
% ============================================================
groups = struct();

for k = 1:nFiles
    fname = files(k).name;
    tok = regexp(fname, '^(sdcq\d+)\#([1-4])\.txt$', 'tokens', 'once');
    if isempty(tok), continue; end

    prefix = tok{1};                 % 例如 sdcq2963
    sid    = str2double(tok{2});     % 1~4
    fpath  = fullfile(files(k).folder, files(k).name);

    if ~isfield(groups, prefix)
        groups.(prefix) = struct( ...
            'hammer','', ...
            'resp',{{'', '', ''}} );
    end

    if sid == 1
        groups.(prefix).hammer = fpath;
    elseif sid >= 2 && sid <= 4
        groups.(prefix).resp{sid-1} = fpath;
    end
end

%% ============================================================
%  3. 将所有测点按编号排序，并重排成 29×64 网格
%     约定：排序后按“先行后列”排列
%     即前64个点为第1行，接着64个点为第2行 ...
% ============================================================
allPrefix = fieldnames(groups);
nPoint = numel(allPrefix);

fprintf('共识别到 %d 个测点前缀。\n', nPoint);

% 提取数字编号，例如 sdcq2963 -> 2963
prefixNum = zeros(nPoint,1);
for i = 1:nPoint
    tok = regexp(allPrefix{i}, '^sdcq(\d+)$', 'tokens', 'once');
    if isempty(tok)
        error('测点前缀格式异常：%s', allPrefix{i});
    end
    prefixNum(i) = str2double(tok{1});
end

% 按数字编号升序排序
[~, idxSort] = sort(prefixNum);
allPrefixSorted = allPrefix(idxSort);

% 检查数量
needPoint = nRow * nCol;
if nPoint < needPoint
    error('测点数量不足：当前仅 %d 个测点，但需要 %d (= %d×%d) 个。', ...
        nPoint, needPoint, nRow, nCol);
elseif nPoint > needPoint
    warning('测点数量多于 %d 个，仅使用排序后的前 %d 个测点进行 %d×%d 重排。', ...
        needPoint, needPoint, nRow, nCol);
    allPrefixSorted = allPrefixSorted(1:needPoint);
end

% 重排为 29行×64列
% 注意：MATLAB reshape 是按列填充，所以这里先 reshape 成 [64,29] 再转置
prefixGrid = reshape(allPrefixSorted, [nCol, nRow]).';

% 取第41列全部测点，共29个
validPrefix = prefixGrid(:, targetCol);

fprintf('已提取第 %d 列，共 %d 个测点。\n', targetCol, numel(validPrefix));
disp(validPrefix);

%% ============================================================
%  4. 并行池
% ============================================================
p = gcp('nocreate');
if ~isempty(p), delete(p); end
parpool('local', 14);

%% ============================================================
%  5. 并行：去卷积 + 作图
% ============================================================
parfor ii = 1:numel(validPrefix)

    prefix = validPrefix{ii};
    rowID  = ii;   % 对应第 ii 行、第 targetCol 列
    G = groups.(prefix);

    % ---------- 力锤 (#1) ----------
    if isempty(G.hammer) || ~isfile(G.hammer)
        fprintf('缺少 hammer 文件：%s\n', prefix);
        continue;
    end

    f = readmatrix(G.hammer);
    f = f(:);
    f = f - mean(f);
    Nf = numel(f);

    % ---------- #2/#3/#4 ----------
    for ch = 1:3
        rpath = G.resp{ch};
        if isempty(rpath) || ~isfile(rpath)
            fprintf('缺少响应文件：%s#%d\n', prefix, ch+1);
            continue;
        end

        y = readmatrix(rpath);
        y = y(:);
        y = y - mean(y);

        % 对齐长度
        N = min(numel(y), Nf);
        f_sig = f(1:N);
        y_sig = y(1:N);

        % ---------------- 频域去卷积 ----------------
        win = hann(N);
        f_sig = f_sig .* win;
        y_sig = y_sig .* win;

        Nfft_use = max(Nfft, 2^nextpow2(N));
        F = fft(f_sig, Nfft_use);
        Y = fft(y_sig, Nfft_use);

        epsReg = alpha * mean(abs(F).^2);
        H = (Y .* conj(F)) ./ (abs(F).^2 + epsReg);

        freq = (0:Nfft_use-1) * fs / Nfft_use;
        mask = (freq >= fband(1) & freq <= fband(2)) | ...
               (freq >= fs-fband(2) & freq <= fs-fband(1));
        H(~mask) = 0;

        h = real(ifft(H, Nfft_use));
        h = h(1:N);

        % ================= 时域图 =================
        t = (0:N-1) / fs;
        fig1 = figure('Visible','off','Units','centimeters', ...
                      'Position',[2 2 8.5 6]);
        plot(t*1e3, h, 'k', 'LineWidth', 1);
        xlabel('Time (ns)','FontName','Times New Roman','FontSize',16);
        ylabel('Amplitude','FontName','Times New Roman','FontSize',16);
        xlim([0 10]);
        grid on;
        set(gca,'FontSize',16,'FontName','Times New Roman', ...
            'LineWidth',1,'Box','on');

        out1 = sprintf('R%02dC%02d_%s#%d_deconv_time.png', ...
            rowID, targetCol, prefix, ch+1);
        print(fig1, fullfile(outputDir, out1), '-dpng','-r300');
        close(fig1);

        % ================= 频域图 =================
        Hh = fft(h, Nfft_use);
        f1 = (0:floor(Nfft_use/2)-1) * fs / Nfft_use;
        Amp = abs(Hh(1:floor(Nfft_use/2))) / N * 2;

        fig2 = figure('Visible','off','Units','centimeters', ...
                      'Position',[2 2 8.5 6]);
        plot(f1/1e3, Amp, 'k', 'LineWidth', 1);
        xlabel('Frequency (kHz)','FontName','Times New Roman','FontSize',16);
        ylabel('Amplitude','FontName','Times New Roman','FontSize',16);
        xlim([1 30]);                 % 频域横坐标范围改为 1~30 kHz
        xticks([1 5 10 15 20 25 30]);
        grid on;
        set(gca,'FontSize',16,'FontName','Times New Roman', ...
            'LineWidth',1,'Box','on');

        out2 = sprintf('R%02dC%02d_%s#%d_deconv_freq.png', ...
            rowID, targetCol, prefix, ch+1);
        print(fig2, fullfile(outputDir, out2), '-dpng','-r300');
        close(fig2);
    end
end

fprintf('\n=========== 去卷积完成：第 %d 列全部测点 ===========\n', targetCol);