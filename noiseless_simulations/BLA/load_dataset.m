function [training_data, test_data, Ts] = load_dataset()
    load("gyroscope_data/ML_estim.mat")
    train_out = exp_y;
    train_in = exp_u;

    load("gyroscope_data/ML_valid.mat");
    test_out = exp_y;
    test_in = exp_u;

    mm = mean(train_out);
    train_out = train_out - mm;
    test_out = test_out - mm;

    mm = mean(train_in);
    train_in = train_in - mm;
    test_in = test_in - mm;

    training_data = iddata(train_in, train_out, Ts);
    test_data = iddata(test_in, test_out, Ts);
end
