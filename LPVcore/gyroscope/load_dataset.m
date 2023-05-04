function [data_est, data_val, Ts] = load_dataset()
    load ML_estim.mat
    exp_y = y_q4d;
    exp_p = [cos(q_all(:, 2)), qd_all(:, 1), sin(q_all(:, 2))];
    exp_u = u_i2;
    data_est = lpviddata(exp_y, exp_p, exp_u, Ts);

    load ML_valid.mat;
    exp_y = y_q4d;
    exp_p = [cos(q_all(:, 2)), qd_all(:, 1), sin(q_all(:, 2))];
    exp_u = u_i2;
    data_val = lpviddata(exp_y, exp_p, exp_u, Ts);
end
