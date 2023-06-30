function [data_train, data_test, Ts] = load_dataset(N)
    load("gyroscope_data/ML_estim.mat")
    exp_y = y_q4d;
    exp_p = [cos(q_all(:, 2)), qd_all(:, 1), sin(q_all(:, 2))];
    exp_u = u_i2;
    if N == 0
        data_train = lpviddata(exp_y, exp_p, exp_u, Ts);
    else
        data_train = lpviddata(exp_y(1:N, :), exp_p(1:N, :), exp_u(1:N, :), Ts);
    end

    load("gyroscope_data/ML_valid.mat");
    exp_y = y_q4d;
    exp_p = [cos(q_all(:, 2)), qd_all(:, 1), sin(q_all(:, 2))];
    exp_u = u_i2;
    data_test = lpviddata(exp_y, exp_p, exp_u, Ts);
end
