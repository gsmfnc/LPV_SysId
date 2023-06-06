function [training_data, test_data, Ts] = load_dataset()
    load("gyroscope_data/ML_estim.mat")
    train_out = y_q4d;
    train_in = u_i2;

    load("gyroscope_data/ML_valid.mat");
    test_out = y_q4d;
    test_in = u_i2;

    mm = mean(train_out);
    train_out = train_out - mm;
    test_out = test_out - mm;

    mm = mean(train_in);
    train_in = train_in - mm;
    test_in = test_in - mm;

    training_data = iddata(train_in, train_out, Ts);
    test_data = iddata(test_in, test_out, Ts);
end
