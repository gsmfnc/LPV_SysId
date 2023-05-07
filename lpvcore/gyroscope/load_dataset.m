function [data_train, data_test, Ts] = load_dataset()
    load("gyroscope_data/ML_estim.mat")
    exp_y = y_q4d;
    exp_p = [cos(q_all(:, 2)), qd_all(:, 1), sin(q_all(:, 2))];
    exp_u = u_i2;
    data_train = lpviddata(exp_y, exp_p, exp_u, Ts);

    load("gyroscope_data/ML_valid.mat");
    exp_y = y_q4d;
    exp_p = [cos(q_all(:, 2)), qd_all(:, 1), sin(q_all(:, 2))];
    exp_u = u_i2;
    data_test = lpviddata(exp_y, exp_p, exp_u, Ts);
end
