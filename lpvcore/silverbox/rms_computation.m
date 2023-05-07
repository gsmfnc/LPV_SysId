function [rms_est, rms_val, rms_test] = rms_computation(est, val, test, ...
        model, plot_)

    rms_est = rms(compare(est, model, Inf) - est.OutputData);
    rms_val = rms(compare(val, model, Inf) - val.OutputData);
    rms_test = rms(compare(test, model, Inf) - test.OutputData);

    if plot_
        figure(); compare(est, model, Inf); title('Training data')
        figure(); compare(val, model, Inf); title('Validation data')
        figure(); compare(test, model, Inf); title('Test data')
    end
end
