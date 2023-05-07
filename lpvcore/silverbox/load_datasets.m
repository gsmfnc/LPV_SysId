function [data1, data2, data3] = load_datasets()
    load('SilverboxFiles/SNLS80mV.mat')
    V1_SNLS = V1;
    V2_SNLS = V2;
    load('SilverboxFiles/Schroeder80mV.mat')
    V1_Schroeder = V1;
    V2_Schroeder = V2;
    tmp_u = V1_SNLS(40650:127400);
    tmp_y = V2_SNLS(40650:127400);
    validation_u = tmp_u(1:floor(size(tmp_u, 2) * 0.25));
    validation_y = tmp_y(1:floor(size(tmp_y, 2) * 0.25));
    training_u = tmp_u((floor(size(tmp_u, 2) * 0.25) + 1):end);
    training_y = tmp_y((floor(size(tmp_y, 2) * 0.25) + 1):end);
    test_u = V1_SNLS(10:40575);
    test_y = V2_SNLS(10:40575);

    training_p = training_y.^2;
    validation_p = validation_y.^2;
    test_p = test_y.^2;

    data1 = lpviddata(training_y(2:end)', training_p(1:(end - 1))', ...
        training_u(2:end)');
    data2 = lpviddata(test_y(2:end)', test_p(1:(end - 1))', test_u(2:end)');
    data3 = lpviddata(validation_y(2:end)', validation_p(1:(end - 1))', ...
        validation_u(2:end)');
end
