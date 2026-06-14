%% ============================================================
% 批量读取 GPR .sgy 文件
% 距离方向统一重采样到 145
% 每个文件：
%   - 单独保存 .mat
%   - 单独保存 .png
%% ============================================================

clear; clc;

%% ------------------ 路径设置 -------------------------------
data_dir = 'C:\GUO\jinqiang\Ultrasonic_GRP\GPR_2';
out_dir_mat = 'C:\GUO\jinqiang\Ultrasonic_GRP\GPR_resample_mat';
out_dir_png = 'C:\GUO\jinqiang\Ultrasonic_GRP\GPR_resample_png';
out_dir_png_color = 'C:\GUO\jinqiang\Ultrasonic_GRP\GPR_resample_png_color';

if ~exist(out_dir_png_color,'dir'), mkdir(out_dir_png_color); end
if ~exist(out_dir_mat,'dir'), mkdir(out_dir_mat); end
if ~exist(out_dir_png,'dir'), mkdir(out_dir_png); end

%% ------------------ 扫描 sgy 文件 ---------------------------
file_list = dir(fullfile(data_dir,'*.sgy'));
Nfile = numel(file_list);

if Nfile == 0
    error('No .sgy files found in %s', data_dir);
end

fprintf('Found %d SGY files.\n', Nfile);

%% ------------------ 目标距离参数 ----------------------------
L_target = 145;                % cm
N_target = 145;                % 145 个距离点
s_target = linspace(0,L_target,N_target);

interp_method = 'pchip';       % 推荐：'pchip' 或 'linear'

%% ------------------ 读取第一个文件，确定 Nt -----------------
fprintf('Detecting Nt from first file...\n');
[Data0,~,~] = ReadSegy(fullfile(data_dir,file_list(1).name));
Nt = size(Data0,1);

fprintf('Nt = %d samples per trace\n', Nt);

%% ------------------ 记录 trace 数 ---------------------------
trace_info = fullfile(out_dir_mat,'trace_count.txt');
fid = fopen(trace_info,'w');
fprintf(fid,'File\tOriginal_trace_count\n');

%% ================== 主循环：逐文件处理 =====================
for i = 1:Nfile
    fname = file_list(i).name;
    fprintf('[%02d/%02d] Processing %s\n', i, Nfile, fname);

    %% ---------- 读取 sgy ----------
    [Data,~,~] = ReadSegy(fullfile(data_dir,fname));
    % Data: Nt x Ntrace

    if size(Data,1) ~= Nt
        error('Nt mismatch in file %s', fname);
    end

    Ntrace = size(Data,2);
    fprintf(fid,'%s\t%d\n',fname,Ntrace);

    %% ---------- 原始距离坐标（不假设130） ----------
    % 将当前 trace 数视为对 145cm 的稀疏采样
    s_old = linspace(0,L_target,Ntrace);

    %% ---------- 插值到 145 ----------
    Data145 = interp1(s_old, Data.', s_target, interp_method, 'extrap').';
    % Data145: Nt x 145

    %% ---------- 保存 .mat ----------
    out_mat = strrep(fname,'.sgy','_resampled.mat');
    save(fullfile(out_dir_mat,out_mat), ...
         'Data145','Nt','Ntrace','L_target','N_target');

    %% ---------- 保存 .png ----------
    fig = figure('Visible','off','Color','w','Position',[100 100 900 600]);
    imagesc(Data145);
    colormap(gray);
    colorbar;
    axis tight;

    xlabel('Distance (cm, resampled)');
    ylabel('Time sample');
    title(strrep(fname,'_','\_'));

    out_png = strrep(fname,'.sgy','_resampled.png');
    exportgraphics(fig, fullfile(out_dir_png,out_png), 'Resolution',300);
    close(fig);

    %% ---------- 额外保存：彩色热力图 .png（新增，不替换灰度） ----------
    figc = figure('Visible','off','Color','w','Position',[100 100 900 600]);
    
    % 建议：用幅值 + dB压缩，更像“蓝底+红黄热点”
    Z = abs(Data145);
    Z = Z ./ (max(Z(:)) + eps);          % 归一化到 [0,1]
    Z = 20*log10(Z + 1e-12);             % dB
    
    imagesc(Z);
    axis tight;
    set(gca,'YDir','reverse');           % 时间向下（常见GPR显示习惯）
    
    colormap(turbo);                     % 推荐：turbo（接近你示例图的效果）
    colorbar;
    caxis([-40 0]);                      % 动态范围：可调 [-60 0] / [-30 0]
    
    xlabel('Distance (cm, resampled)');
    ylabel('Time sample');
    title(strrep(fname,'_','\_'));
    
    out_png_c = strrep(fname,'.sgy','_resampled_color.png');
    exportgraphics(figc, fullfile(out_dir_png_color,out_png_c), 'Resolution',300);
    close(figc);

end

fclose(fid);

fprintf('\nAll files processed successfully.\n');
fprintf('MAT saved to : %s\n', out_dir_mat);
fprintf('PNG saved to : %s\n', out_dir_png);
