%% ========= Plot A-scan (0-6 ns) WITHOUT normalization =========
clear; clc; close all;

% -------- input --------
in_file = 'C:\GUO\jinqiang\Ultrasonic_GRP\GPR_resample_mat\GPR_41_resampled.mat';

idx_intact = 2;      % intact point (column index)
idx_Void = 111;    % Void point (column index)

T_window_ns = 16;    % full time window in file (ns)
t_zoom = [0 6];      % plot window (ns)

% -------- output: new folder each run --------
root_out = 'C:\GUO\jinqiang\Ultrasonic_GRP\fig_ascan_outputs';
if ~exist(root_out,'dir'), mkdir(root_out); end
out_dir = fullfile(root_out, ['run_' datestr(now,'yyyymmdd_HHMMSS')]);
if ~exist(out_dir,'dir'), mkdir(out_dir); end

% -------- load --------
S = load(in_file);

% Data145: Nt × 145 (each column = one A-scan)
X = S.Data145;

% robust orientation check
if size(X,2) < size(X,1) && size(X,1) == 145
    % sometimes stored as 145 × Nt
    X = X.';
end

Nt = size(X,1);
Ncol = size(X,2);

% sanity check
if idx_intact < 1 || idx_intact > Ncol || idx_Void < 1 || idx_Void > Ncol
    error('Column index out of range. Data has %d columns, but idx_intact=%d, idx_Void=%d.', ...
        Ncol, idx_intact, idx_Void);
end

t_ns = linspace(0, T_window_ns, Nt);
mask = (t_ns >= t_zoom(1)) & (t_ns <= t_zoom(2));

xI = double(X(:, idx_intact));
xD = double(X(:, idx_Void));

%% -------- Fig 1: raw A-scan (time-domain), NO title, fixed y-limits --------
fig1 = figure('Color','w','Position',[120 120 600 320]);
ax1 = axes(fig1); hold(ax1,'on'); box(ax1,'on');

plot(ax1, t_ns(mask), xI(mask), 'k-',  'LineWidth',1.5);
plot(ax1, t_ns(mask), xD(mask), 'k--', 'LineWidth',1.5);

xlabel(ax1, 'Two-way travel time, t (ns)', 'FontName','Times New Roman','FontSize',18);
ylabel(ax1, 'Amplitude',           'FontName','Times New Roman','FontSize',18);

legend(ax1, {'Intact A-scan','Void A-scan'}, 'Location','northeast','Box','off');
set(ax1,'FontName','Times New Roman','FontSize',18,'LineWidth',1.0);

xlim(ax1, t_zoom);

% ★ 强制固定纵轴范围（放在最后，并设为手动模式）
ylim(ax1, [-0.03 0.03]);
set(ax1, 'YLimMode','manual');
yt = -0.03:0.01:0.03;
yticks(ax1, yt);
yticklabels(ax1, compose('%.2f', yt));  % 显示 -0.025 / 0.000 / 0.025

exportgraphics(fig1, fullfile(out_dir,'Fig_Ascan_raw_0_6ns.png'), 'Resolution',600);
exportgraphics(fig1, fullfile(out_dir,'Fig_Ascan_raw_0_6ns.pdf'), 'ContentType','vector');

%% -------- Fig 2: envelope only (Hilbert envelope), 0-6 ns --------
envI = abs(hilbert(xI));
envD = abs(hilbert(xD));

fig2 = figure('Color','w','Position',[140 140 600 320]);
ax2 = axes(fig2); hold(ax2,'on'); box(ax2,'on');

plot(ax2, t_ns(mask), envI(mask), 'LineWidth',2.0);
plot(ax2, t_ns(mask), envD(mask), 'LineWidth',2.0);

xlabel(ax2, 'Two-way travel time, t (ns)', 'FontName','Times New Roman','FontSize',18);
ylabel(ax2, 'Envelope amplitude',   'FontName','Times New Roman','FontSize',18);


% title(ax2, sprintf('Hilbert envelopes (0–%g ns): intact (col %d) vs Void (col %d)', ...
%       t_zoom(2), idx_intact, idx_Void), ...
%       'FontName','Times New Roman','FontSize',14,'FontWeight','normal');

legend(ax2, {'Intact envelope','Void envelope'}, 'Location','northeast','Box','off');
set(ax2,'FontName','Times New Roman','FontSize',18,'LineWidth',1.0);

xlim(ax2, t_zoom);
ylim(ax2, [0 0.03]);
set(ax2,'YLimMode','manual');

% 纵坐标刻度保留三位小数（可调步长）
yt2 = 0:0.01:0.03;             % 你想更稀疏可改成 0:0.01:0.03
yticks(ax2, yt2);

exportgraphics(fig2, fullfile(out_dir,'Fig_Ascan_envelope_0_6ns.png'), 'Resolution',600);
exportgraphics(fig2, fullfile(out_dir,'Fig_Ascan_envelope_0_6ns.pdf'), 'ContentType','vector');

fprintf('Saved figures to:\n%s\n', out_dir);