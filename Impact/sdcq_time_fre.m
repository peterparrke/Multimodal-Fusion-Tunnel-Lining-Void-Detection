clc; clear; close all;

%% ======================= 参数设置 ============================
inputDir  = 'C:\GUO\jinqiang\data\sdcq_2';
outputDir = 'C:\GUO\jinqiang\data_viasual\time_fre';
if ~exist(outputDir, 'dir'), mkdir(outputDir); end

fs   = 1e6;        % 采样率 (Hz)
Nfft = 131072;     % FFT 点数

files = dir(fullfile(inputDir, 'sdcq*.txt'));

fprintf("共找到 %d 个单列信号文件。\n", length(files));

idxStart = 5377;
idxEnd   = 5632;

%% ======================= 启动并行池 ============================
p = gcp('nocreate');
if ~isempty(p)
    delete(p);
end
parpool('local', 14);

%% ======================= 并行处理 ============================
parfor i = idxStart:idxEnd

    %% -------- 1. 读取数据 ----------
    filePath = fullfile(files(i).folder, files(i).name);
    x = readmatrix(filePath);
    x = x(:,2);            % 使用第 2 列
    nt = length(x);

    %% -------- 2. 去直流 ----------
    x = x - mean(x);

    %% -------- 3. FFT ----------
    Y = fft(x, Nfft);
    f = (0:Nfft/2-1) * fs / Nfft;
    Amp = abs(Y(1:Nfft/2)) / nt * 2;

    t = (0:nt-1) / fs;

    %% 文件名
    [~, baseName, ~] = fileparts(files(i).name);

    %% =========================================================
    %                    时域图（单独）
    % =========================================================
    fig1 = figure('Visible','off','Units','centimeters',...
                  'Position',[2 2 8.5 6]);   % 单栏友好

    plot(t*1e3, x, 'k', 'LineWidth', 1);   % 黑色
    xlabel('Time (ms)', 'FontName','Times New Roman','FontSize',10);
    ylabel('Amplitude', 'FontName','Times New Roman','FontSize',10);
    xlim([0 10]);
    grid on;
    set(gca,'FontSize',9, 'FontName','Times New Roman', 'LineWidth',1,'Box','on');

    print(fig1, fullfile(outputDir, [baseName '_time.png']), ...
          '-dpng','-r300');
    close(fig1);

    %% =========================================================
    %                    频域图（单独）
    % =========================================================
    fig2 = figure('Visible','off','Units','centimeters',...
                  'Position',[2 2 8.5 6]);

    plot(f/1e3, Amp, 'k', 'LineWidth', 1);
    xlabel('Frequency (kHz)', 'FontName','Times New Roman','FontSize',10);
    ylabel('Amplitude', 'FontName','Times New Roman','FontSize',10);
    xlim([0 40]);
    grid on;
    set(gca,'FontSize',9, 'FontName','Times New Roman','LineWidth',1,'Box','on');

    print(fig2, fullfile(outputDir, [baseName '_freq.png']), ...
          '-dpng','-r300');
    close(fig2);

    fprintf("完成：%s\n", baseName);
end

fprintf("\n=========== 时域 / 频域图像全部生成完毕！===========\n");
