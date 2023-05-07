function [rms_est, rms_test] = rms_computation(est, test, model, ...
        plot_)

    rms_est = rms(compare(est, model, Inf) - est.OutputData);
    rms_test = rms(compare(test, model, Inf) - test.OutputData);

    if plot_
        figure(); compare(est, model, Inf); title('Training data')
        figure(); compare(test, model, Inf); title('Test data')
    end
end
