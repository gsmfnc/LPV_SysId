function [rms_est, rms_test] = rms_computation(est, test, model)
    rms_est = rms(compare(est, model, Inf) - est.OutputData);
    rms_test = rms(compare(test, model, Inf) - test.OutputData);
end
