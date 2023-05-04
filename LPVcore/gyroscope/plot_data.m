function plot_data(plot_, data_est, data_val)
    if plot_
        figure(1)
        subplot(211); plot(data_est.OutputData)
        title('Output estimation data')
        subplot(212); plot(data_val.OutputData)
        title('Output validation data')

        figure(2)
        subplot(321); plot(data_est.SchedulingData(:, 1))
        title('Scheduling 1 estimation data')
        subplot(322); plot(data_val.SchedulingData(:, 1))
        title('Scheduling 1 validation data')
        subplot(323); plot(data_est.SchedulingData(:, 2))
        title('Scheduling 2 estimation data')
        subplot(324); plot(data_val.SchedulingData(:, 2))
        title('Scheduling 2 validation data')
        subplot(325); plot(data_est.SchedulingData(:, 3))
        title('Scheduling 3 estimation data')
        subplot(326); plot(data_val.SchedulingData(:, 3))
        title('Scheduling 3 validation data')

        figure(3)
        subplot(211); plot(data_est.InputData)
        title('Input estimation data')
        subplot(212); plot(data_val.InputData)
        title('Input validation data')
    end
end
