close all
fsize = 18;

fig1 = figure(1)
[model_output, fit] = compare(data_test, arx_model, Inf);
plot(data_test.OutputData); hold on
plot(model_output - data_test.OutputData);
legend('$y$', '$y-\hat y$', 'interpreter', 'latex', 'fontsize', fsize)
bfr = compute_bfr(model_output, data_test.OutputData);
title("BFR=" + bfr + "%", 'interpreter', 'latex', 'fontsize', fsize)

fig2 = figure(2)
[model_output, fit] = compare(data_test, oe_model, Inf);
plot(data_test.OutputData); hold on
plot(model_output - data_test.OutputData);
legend('$y$', '$y-\hat y$', 'interpreter', 'latex', 'fontsize', fsize)
bfr = compute_bfr(model_output, data_test.OutputData);
title("BFR=" + bfr + "%", 'interpreter', 'latex', 'fontsize', fsize)

fig3 = figure(3)
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
