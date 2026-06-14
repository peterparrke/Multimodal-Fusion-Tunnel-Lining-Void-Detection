clc; clear; close all;

IN_DIR = 'C:\GUO\jinqiang\Ultrasonic_GRP\GPR_resample_mat';
PARENT_DIR = fileparts(IN_DIR);
OUT_DIR = fullfile(PARENT_DIR, 'GPR_Cscan_color_out');
if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

TIME_WINDOWS = { ...
    'win_030_050', 30, 50; ...
    'win_050_070', 50, 70; ...
    'win_070_090', 70, 90; ...
    'win_090_110', 90, 110; ...
    'win_110_130', 110, 130; ...
    'win_130_150', 130, 150; ...
    'win_150_170', 150, 170; ...
    'win_170_190', 170, 190; ...
    'win_190_210', 190, 210; ...
    'win_210_230', 210, 230; ...
};

USE_BG_REMOVAL = true;
USE_TIME_GAIN  = false;
GAIN_MAX       = 6;
MUTE_EARLY     = 0;

VAR_NAME = '';              % 留空自动找最大的2D数值矩阵
DO_MAX = true;
DO_RMS = true;

UPSAMPLE_METHOD   = 'linear';  % 'repeat' or 'linear'
UPSAMPLE_X_FACTOR = 5;         % 64 -> 320

DY_CM = 1;                  % 竖向 1 cm
DX_CM = 5;                  % 横向原始 5 cm

% 固定颜色范围（所有图一致）
FIX_CLIM = true;
CLIM = [0, 3e-2];         % <- 按你数据调整

% 绘图更像GPR：用 surface + shading interp
USE_GPR_LOOK = true;

files = dir(fullfile(IN_DIR, 'GPR_*_resampled.mat'));
if isempty(files)
    error('未找到文件：%s', fullfile(IN_DIR, 'GPR_*_resampled.mat'));
end

idx = zeros(numel(files),1);
for i = 1:numel(files)
    tk = regexp(files(i).name, 'GPR_(\d+)', 'tokens', 'once');
    if isempty(tk), idx(i)=1e9; else, idx(i)=str2double(tk{1}); end
end
[~, ord] = sort(idx);
files = files(ord);

Nx = numel(files);

Dcell = cell(Nx,1);
Nt_list = zeros(Nx,1);
Ny_list = zeros(Nx,1);

for ix = 1:Nx
    fp = fullfile(files(ix).folder, files(ix).name);
    S = load(fp);
    D = pick_2d_matrix(S, VAR_NAME);
    D = double(D);

    % 统一为 (Nt, Ny)
    if size(D,1) < size(D,2)
        D = D.';
    end

    if MUTE_EARLY > 0
        D(1:min(MUTE_EARLY,size(D,1)), :) = 0;
    end

    if USE_BG_REMOVAL
        D = D - mean(D, 2);
    end

    if USE_TIME_GAIN
        Nt0 = size(D,1);
        gain = linspace(1, GAIN_MAX, Nt0).';
        D = D .* gain;
    end

    Dcell{ix} = D;
    Nt_list(ix) = size(D,1);
    Ny_list(ix) = size(D,2);
end

Nt = min(Nt_list);
Ny = min(Ny_list);

Vol = zeros(Nt, Ny, Nx);
for ix = 1:Nx
    D = Dcell{ix};
    Vol(:,:,ix) = D(1:Nt, 1:Ny);
end
save(fullfile(OUT_DIR, 'gpr_volume_raw.mat'), 'Vol', '-v7.3');

X = reshape(Vol, Nt, []);
Env = abs(hilbert(X));
EnvVol = reshape(Env, Nt, Ny, Nx);
save(fullfile(OUT_DIR, 'gpr_volume_env.mat'), 'EnvVol', '-v7.3');

for k = 1:size(TIME_WINDOWS,1)
    tag = TIME_WINDOWS{k,1};
    t1  = max(1, TIME_WINDOWS{k,2});
    t2  = min(Nt, TIME_WINDOWS{k,3});
    if t2 <= t1, continue; end

    slab = EnvVol(t1:t2, :, :);

    if DO_MAX
        Cmax = squeeze(max(slab, [], 1));  % (Ny, Nx)
        Cmax_up = upsample_x(Cmax, UPSAMPLE_X_FACTOR, UPSAMPLE_METHOD);

        save(fullfile(OUT_DIR, ['cscan_' tag '_MAX.mat']), 'Cmax', 'Cmax_up', 't1', 't2');

        export_cscan_color(Cmax_up, fullfile(OUT_DIR, ['cscan_' tag '_MAX.png']), ...
            sprintf('C-scan MAX envelope, t=[%d,%d]', t1, t2), ...
            DY_CM, DX_CM/UPSAMPLE_X_FACTOR, FIX_CLIM, CLIM, USE_GPR_LOOK);
    end

    if DO_RMS
        Crms = squeeze(sqrt(mean(slab.^2, 1)));  % (Ny, Nx)
        Crms_up = upsample_x(Crms, UPSAMPLE_X_FACTOR, UPSAMPLE_METHOD);

        save(fullfile(OUT_DIR, ['cscan_' tag '_RMS.mat']), 'Crms', 'Crms_up', 't1', 't2');

        export_cscan_color(Crms_up, fullfile(OUT_DIR, ['cscan_' tag '_RMS.png']), ...
            sprintf('C-scan RMS envelope, t=[%d,%d]', t1, t2), ...
            DY_CM, DX_CM/UPSAMPLE_X_FACTOR, FIX_CLIM, CLIM, USE_GPR_LOOK);
    end
end

fprintf('DONE: %s\n', OUT_DIR);

% ---------------- functions ----------------
function D = pick_2d_matrix(S, varName)
    if ~isempty(varName)
        if ~isfield(S, varName)
            f = fieldnames(S);
            error('变量名 "%s" 不存在。该mat变量：%s', varName, strjoin(f.', ', '));
        end
        D = S.(varName);
        if ~(isnumeric(D) && ismatrix(D))
            error('变量 "%s" 不是2D数值矩阵。', varName);
        end
        return;
    end

    fn = fieldnames(S);
    bestSize = -inf; bestName = '';
    for i = 1:numel(fn)
        v = S.(fn{i});
        if isnumeric(v) && ismatrix(v)
            if numel(v) > bestSize
                bestSize = numel(v);
                bestName = fn{i};
            end
        end
    end
    if bestSize < 0
        error('未找到任何2D数值矩阵。');
    end
    D = S.(bestName);
end

function Cup = upsample_x(C, factor, method)
    if factor <= 1
        Cup = C; return;
    end
    [Ny, Nx] = size(C);
    switch lower(method)
        case 'repeat'
            Cup = repelem(C, 1, factor);
        case 'linear'
            x0 = 1:Nx;
            xq = linspace(1, Nx, Nx*factor);
            Cup = zeros(Ny, numel(xq));
            for r = 1:Ny
                Cup(r,:) = interp1(x0, C(r,:), xq, 'linear');
            end
        otherwise
            error('UPSAMPLE_METHOD 只能是 repeat 或 linear');
    end
end

function export_cscan_color(C, out_png, ttl, dy_cm, dx_cm, FIX_CLIM, CLIM, USE_GPR_LOOK)
    C = double(C);
    [Ny, Nx] = size(C);
    x_cm = (0:Nx-1) * dx_cm;
    y_cm = (0:Ny-1) * dy_cm;

    fig = figure('Visible','off','Color','w');

    if USE_GPR_LOOK
        [X, Y] = meshgrid(x_cm, y_cm);
        h = surf(X, Y, zeros(size(C)), C);
        set(h, 'EdgeColor', 'none');
        view(2);
        shading interp; % 更像GPR软件的渲染
    else
        imagesc(x_cm, y_cm, C);
    end

    set(gca,'YDir','reverse');
    axis image;

    if FIX_CLIM
        caxis(CLIM);
    else
        vmin = prctile(C(:), 2);
        vmax = prctile(C(:), 98);
        if vmax <= vmin
            vmin = min(C(:)); vmax = max(C(:));
        end
        caxis([vmin vmax]);
    end

    colorbar;
    title(ttl, 'Interpreter','none');
    xlabel('X (cm)');
    ylabel('Y (cm)');
    set(gca,'FontSize',11);

    if exist('turbo', 'file')
        colormap(turbo);
    else
        colormap(jet);
    end

    exportgraphics(fig, out_png, 'Resolution', 300);
    close(fig);
end