function plot_data(plot_, data_train, data_val, data_test)
    if plot_
        figure(1)
        subplot(311); plot(data_train.OutputData)
        title('Output training data')
        subplot(312); plot(data_val.OutputData)
        title('Output validation data')
        subplot(313); plot(data_test.OutputData)
        title('Output test data')

        figure(2)
        subplot(311); plot(data_train.SchedulingData(:, 1))
        title('Scheduling training data')
        subplot(312); plot(data_val.SchedulingData(:, 1))
        title('Scheduling validation data')
        subplot(313); plot(data_test.SchedulingData(:, 1))
        title('Scheduling test data')

        figure(3)
        subplot(311); plot(data_train.InputData)
        title('Input training data')
        subplot(312); plot(data_val.InputData)
        title('Input validation data')
        subplot(313); plot(data_test.InputData)
        title('Input test data')
    end
end
