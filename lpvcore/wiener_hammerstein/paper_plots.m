fig = figure(1)
fsize = 18;
[model_output, fit] = compare(data_test, oe_model, Inf);
plot(data_test.OutputData); hold on
plot(model_output - data_test.OutputData);
legend('$y$', '$y-\hat y$', 'interpreter', 'latex', 'fontsize', fsize)
title("BFR=" + fit + "%", 'interpreter', 'latex')
