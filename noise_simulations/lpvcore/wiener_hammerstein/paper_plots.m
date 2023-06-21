close all
set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

fig = figure(1)
fsize = 18;
tsize = 12;
[model_output, fit] = compare(data_test, oe_model, Inf);
plot(data_test.OutputData); hold on
plot(model_output - data_test.OutputData);
legend('$y$', '$y-\hat y$', 'interpreter', 'latex', 'fontsize', fsize)
bfr = compute_bfr(model_output, data_test.OutputData);
title("BFR=" + bfr + "%", 'interpreter', 'latex', 'fontsize', fsize)

fig2 = figure(2)
fsize = 18;
tsize = 12;
p1 = subplot(221);
plot(data_train.InputData); title('Training set', 'fontsize', tsize)
ylabel('$u$', 'interpreter', 'latex', 'fontsize', fsize)
set(gca, 'xticklabel', []);
yticks([-2, -1, 0, 1, 2, 3])
p2 = subplot(222);
plot(data_test.InputData); title('Test set', 'fontsize', tsize)
set(gca, 'xticklabel', []);
yticks([-3, -2, -1, 0, 1, 2, 3])
p3 = subplot(223);
plot(data_train.OutputData);
xlabel('$k$', 'interpreter', 'latex', 'fontsize', fsize)
ylabel('$y$', 'interpreter', 'latex', 'fontsize', fsize)
yticks([-1.2, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4])
p4 = subplot(224);
plot(data_test.OutputData);
xlabel('$k$', 'interpreter', 'latex', 'fontsize', fsize)
yticks([-1.2, -1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])
set(p1, 'position', [.1 .5 .4 .45])
set(p2, 'position', [.55 .5 .4 .45])
set(p3, 'position', [.1 .05 .4 .45])
set(p4, 'position', [.55 .05 .4 .45])

fi3 = figure(3)
fsize = 18;
tsize = 12;
[model_output, fit] = compare(data_test, arx_model, Inf);
plot(data_test.OutputData); hold on
plot(model_output - data_test.OutputData);
legend('$y$', '$y-\hat y$', 'interpreter', 'latex', 'fontsize', fsize)
bfr = compute_bfr(model_output, data_test.OutputData);
title("BFR=" + bfr + "%", 'interpreter', 'latex', 'fontsize', fsize)

fig4 = figure(4)
fsize = 18;
tsize = 12;
[model_output, fit] = compare(data_test, ss_model, Inf);
plot(data_test.OutputData); hold on
plot(model_output - data_test.OutputData);
legend('$y$', '$y-\hat y$', 'interpreter', 'latex', 'fontsize', fsize)
bfr = compute_bfr(model_output, data_test.OutputData);
title("BFR=" + bfr + "%", 'interpreter', 'latex', 'fontsize', fsize)

function bfr = compute_bfr(y, yr)
num = 0;
den = 0;
mean_y = mean(yr);
num = norm(y - yr);
den = norm(yr - mean_y);

bfr = max(1 - num / den, 0) * 100
end
