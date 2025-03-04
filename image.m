% 设定六个数据集
ds_list = {
    [16, 32, 64, 128, 256], 
    [16, 32, 64, 128, 256], 
    [16, 32, 64, 128, 256], 
    [1, 2, 3, 4, 5], 
    [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],  
    [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
};

metrics = { 
    % Precision, Recall, NDCG
    [0.04953, 0.05234, 0.05672, 0.05422, 0.0516;
     0.48333, 0.50729, 0.55339, 0.5276, 0.5037;
     0.31412, 0.35769, 0.37845, 0.36352, 0.34975],



    [0.03078, 0.04484, 0.05028, 0.05672, 0.04875;
     0.30286, 0.42916, 0.49849, 0.55339, 0.47552;
     0.15787, 0.27552, 0.3476, 0.37845, 0.33919],

     [0.05016, 0.05515, 0.05672, 0.04984, 0.04688;
     0.49063, 0.53776, 0.55339, 0.48593, 0.45469;
     0.34321, 0.36122, 0.37845, 0.33003, 0.32208],
     
     
    [0.05141, 0.05672, 0.05246, 0.04956, 0.04844;
     0.49974, 0.55339, 0.5056, 0.47682, 0.46796;
     0.3441, 0.37845, 0.35048, 0.32734, 0.3282],

     
    [0.05031, 0.05188, 0.05313, 0.05672, 0.04672;
     0.48489, 0.50339, 0.51849, 0.55339, 0.45313;
     0.32707, 0.34158, 0.36525, 0.37845, 0.28716]
     
    [0.04813, 0.05281, 0.05672, 0.05375, 0.04969;
     0.46589, 0.5138, 0.55339, 0.51549, 0.48008;
     0.31601, 0.36787, 0.37845, 0.3434, 0.3105],

};

xlabels = {
    '\( d_l \)', 
    '\( d_s \)', 
    '\( d_g \)', 
    '\( d_k \)', 
     '\( \lambda_1 \)',
    '\( \lambda_2 \)', 

};

titles = {
    '(a) The effect of $d_l$', 
    '(b) The effect of $d_s$', 
    '(c) The effect of $d_g$', 
    '(d) The effect of $d_k$', 
    '(e) The effect of \( \lambda_1 \)',
    '(f) The effect of \( \lambda_2 \)' 

};

% 设定画布大小 (30x40 cm)，减少白边
figure('Units', 'centimeters', 'Position', [5, 5, 23, 35]);

% 设定子图布局参数
rows = 3;
cols = 2;

left_margin = 0.04;  % 左边距
bottom_margin = 0.03; % 底边距
plot_width = 0.43;    % 每个子图的宽度
plot_height = 0.25;   % 每个子图的高度
x_spacing = 0.08;    % 水平间距
y_spacing = 0.07;    % 垂直间距

for i = 1:6
    row = floor((i-1) / cols);  % 计算当前子图所在的行
    col = mod(i-1, cols);       % 计算当前子图所在的列

    % 计算子图的 Position（[left, bottom, width, height]）
    left = left_margin + col * (plot_width + x_spacing);
    bottom = 1 - (row + 1) * (plot_height + y_spacing);

    % 创建子图，并设置 Position
    ax = subplot(rows, cols, i);
    set(ax, 'Position', [left, bottom, plot_width, plot_height]);

    % 数据处理
    x_new = 1:length(ds_list{i});
    precision = metrics{i}(1, :);
    recall = metrics{i}(2, :);
    f1_scores = 2 * (precision .* recall) ./ (precision + recall);

    % 画图
    plot(x_new, precision, 'g-o', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Precision'); hold on;
    plot(x_new, recall, 'y-s', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'Recall','Color', [0.988, 0.957, 0.012]);
    plot(x_new, f1_scores, 'r-d', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'F1 Score');
    plot(x_new, metrics{i}(3, :), 'b-x', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', 'NDCG');

    ylim([0, 0.8]);
    set(gca, 'XTick', x_new);

    % 处理 X 轴标签
    if i == 5 || i == 6
        set(gca, 'XTickLabel', {'$10^{-1}$', '$10^{-2}$', '$10^{-3}$', '$10^{-4}$', '$10^{-5}$'}, 'TickLabelInterpreter', 'latex');
    else
        set(gca, 'XTickLabel', ds_list{i});
    end

    title(titles{i}, 'Interpreter', 'latex', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel(xlabels{i}, 'Interpreter', 'latex', 'FontSize', 13);
    legend('show', 'Location', 'northwest');
    grid on;

    % 调整 x 轴范围，避免数据点贴边
    padding = 0.1;
    x_range = max(x_new) - min(x_new);
    xlim([min(x_new) - padding * x_range, max(x_new) + padding * x_range]);
end

% 导出图片
print(gcf, 'output.png', '-dpng', '-r300');
