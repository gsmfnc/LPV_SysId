function [data_LPV_training, data_LPV_test] = load_datasets()
    load('WienerHammerBenchMark.mat')

    tmp_u = uBenchMark(5200:184000);
    tmp_y = yBenchMark(5200:184000);

    training_u = tmp_u(2:100000);
    training_y = tmp_y(2:100000);
    test_u = tmp_u(100002:end);
    test_y = tmp_y(100002:end);

    train_tmp = tmp_y(1:99999);
    test_tmp = tmp_y(100001:(end - 1)); 
    training_p = [train_tmp train_tmp.^2 train_tmp.^3];
    test_p = [test_tmp test_tmp.^2 test_tmp.^3]; 

    data_LPV_training = lpviddata(training_y, training_p, training_u);
    data_LPV_test = lpviddata(test_y, test_p, test_u);
end
