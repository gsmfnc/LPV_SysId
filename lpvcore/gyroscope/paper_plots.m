close all
fig = figure(1)
fsize = 18;
subplot(421);
plot(data_train.InputData); title('Training set', 'fontsize', fsize)
ylabel('$i_2$ (A)', 'interpreter', 'latex', 'fontsize', fsize)
subplot(422);
plot(data_test.InputData); title('Test set', 'fontsize', fsize)
subplot(423);
plot(data_train.SchedulingData(:, 1)); hold on
plot(data_train.SchedulingData(:, 3));
legend('$\cos(q_2)$', '$\sin(q_2)$', 'interpreter', 'latex', 'fontsize', fsize)
subplot(424);
plot(data_test.SchedulingData(:, 1)); hold on
plot(data_test.SchedulingData(:, 3));
legend('$\cos(q_2)$', '$\sin(q_2)$', 'interpreter', 'latex', 'fontsize', fsize)
subplot(425);
plot(data_train.SchedulingData(:, 2));
ylabel('$\dot q_1$ (rad/s)', 'interpreter', 'latex', 'fontsize', fsize)
subplot(426);
plot(data_test.SchedulingData(:, 2));
subplot(427);
plot(data_train.OutputData);
ylabel('$\dot q_4$ (rad/s)', 'interpreter', 'latex', 'fontsize', fsize)
subplot(428);
plot(data_test.OutputData);

fig2 = figure(2)
[model_output, fit] = compare(data_test, oe_model, Inf);
plot(data_test.OutputData); hold on
plot(model_output - data_test.OutputData);
legend('$y$', '$y-\hat y$', 'interpreter', 'latex', 'fontsize', fsize)
title("BFR=" + fit + "%", 'interpreter', 'latex')
