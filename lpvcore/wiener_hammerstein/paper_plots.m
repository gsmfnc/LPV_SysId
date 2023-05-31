close all
fig = figure(1)
fsize = 18;
[model_output, fit] = compare(data_test, oe_model, Inf);
plot(data_test.OutputData); hold on
plot(model_output - data_test.OutputData);
legend('$y$', '$y-\hat y$', 'interpreter', 'latex', 'fontsize', fsize)
bfr = compute_bfr(model_output, data_test.OutputData);
title("BFR=" + bfr + "%", 'interpreter', 'latex', 'fontsize', fsize)

fig2 = figure(2)
fsize = 18;
subplot(221);
plot(data_train.InputData); title('Training set', 'fontsize', fsize)
ylabel('$u$', 'interpreter', 'latex', 'fontsize', fsize)
subplot(222);
plot(data_test.InputData); title('Test set', 'fontsize', fsize)
subplot(223);
plot(data_train.OutputData);
ylabel('$y$', 'interpreter', 'latex', 'fontsize', fsize)
subplot(224);
plot(data_test.OutputData);

function bfr = compute_bfr(y, yr)
num = 0;
den = 0;
mean_y = mean(yr);
num = norm(y - yr);
den = norm(yr - mean_y);

bfr = max(1 - num / den, 0) * 100
end
