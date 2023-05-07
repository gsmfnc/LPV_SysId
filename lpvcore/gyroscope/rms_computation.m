function [rms_est, rms_val] = rms_computation(est, val, model, plot_)

    rms_est = rms(compare(est, model, Inf) - est.OutputData);
    rms_val = rms(compare(val, model, Inf) - val.OutputData);

    if plot_
        figure(); compare(est, model, Inf);
        figure(); compare(val, model, Inf);
    end
end
