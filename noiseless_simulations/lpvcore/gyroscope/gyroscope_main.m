clc
clear
close all

value_for_ZeroIsNonFree = false;
plot_ = false;
training_ = false;

[data_train, data_test, Ts] = load_dataset();
plot_data(plot_, data_train, data_test)

%% LPV-ARX estimation
s2 = preal('sin(q2)', 'dt');
c2 = preal('cos(q2)', 'dt');
q1d = preal('q1d', 'dt');

A = randn(1) + randn(1) * q1d + randn(1) * s2 + randn(1) * c2;
B = randn(1) + randn(1) * q1d + randn(1) * s2 + randn(1) * c2;
na = 5;
nb = 5;
[A_poly, B_poly] = shift_pol(A, na, B, nb);

template_arx = lpvidpoly(A_poly, B_poly, [], [], [], 0, Ts, ...
    'ZeroIsNonFree', value_for_ZeroIsNonFree);
template_arx.InputName = {'Current Gimbal 2'};
template_arx.InputUnit = {'A'};
template_arx.OutputName = {'q4d'};
template_arx.OutputUnit = {'rad/s'};
if training_
    arx_model = lpvarx(data_train, template_arx);
else
    load gyroscope_arx_model
end
[rms_est, rms_test] = rms_computation(data_train, data_test, arx_model, plot_)

%% LPV-OE estimation
template_oe = lpvidpoly([], arx_model.B, [], [], arx_model.A, 0, Ts, ...
    'ZeroIsNonFree', value_for_ZeroIsNonFree);
%template_oe = lpvidpoly([], B_poly, [], [], A_poly, 0, Ts, ...
%    'ZeroIsNonFree',value_for_ZeroIsNonFree);
template_oe.InputName = {'Current Gimbal 2'};
template_oe.InputUnit = {'A'};
template_oe.OutputName = {'q4d'};
template_oe.OutputUnit = {'rad/s'};
options_oe = lpvoeOptions('Display', 'off', ...
    'SearchOptions',  struct('StepSize', 10, 'StepTolerance', 1E-10), ...
    'Regularization', struct('Lambda', 1),...
    'Initialization', 'template');
if training_
    oe_model = lpvoe(data_train, template_oe, options_oe);
else
    load gyroscope_oe_model
end
[rms_est, rms_test] = rms_computation(data_train, data_test, oe_model, plot_)

%% LPV-SS estimation
template_ss = lpvio2ss(oe_model.F, oe_model.B, na, nb, Ts, {q1d, s2, c2});
template_ss.Ts = Ts;
options_pem_ss = lpvssestOptions;
options_pem_ss.Display = 'off';
options_pem_ss.Initialization = 'template';
if training_
    ss_model = lpvssest(data_train, template_ss, options_pem_ss);
else
    load gyroscope_ss_model
end
[rms_est, rms_test] = rms_computation(data_train, data_test, ss_model, plot_)
