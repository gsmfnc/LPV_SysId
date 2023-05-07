clc
clear
close all

value_for_ZeroIsNonFree = false;
plot_ = true;
training_ = false;

[data_train, data_test, data_val] = load_datasets();
plot_data(plot_, data_train, data_val, data_test)

%% LPV-ARX estimation
yp = preal('y', 'dt');

A = randn(1) + randn(1) * yp;
B = randn(1) + randn(1) * yp;
na = 2;
nb = 2;
[A_poly, B_poly] = shift_pol(A, na, B, nb);

template_arx = lpvidpoly(A_poly, B_poly, [], [], [], 0, ...
    'ZeroIsNonFree', value_for_ZeroIsNonFree);
options = lpvarxOptions('Display', 'off');
if training_
    arx_model = lpvarx(data_train, template_arx, options);
else
    load silverbox_arx_model
end
[rms_train, rms_val, rms_test] = rms_computation(data_train, data_val, ...
    data_test, arx_model, plot_)

%% LPV-OE estimation
template_oe = lpvidpoly([], arx_model.B, [], [], arx_model.A, 0, ...
    'ZeroIsNonFree', value_for_ZeroIsNonFree);
options_oe = lpvoeOptions('Display', 'off', ...
    'SearchOptions', struct('StepSize', 10, 'StepTolerance', 1E-10), ...
    'Regularization', struct('Lambda', 1),...
    'Initialization', 'template');
if training_
    oe_model = lpvoe(data_train, template_oe, options_oe);
else
    load silverbox_oe_model
end
[rms_train, rms_val, rms_test] = rms_computation(data_train, data_val, ...
    data_test, oe_model, plot_)

%% LPV-SS estimation
%template_ss = lpvio2ss(oe_model.F, oe_model.B, na, nb, 1, {yp});
%options_pem_ss = lpvssestOptions;
%options_pem_ss.Display = 'off';
%options_pem_ss.Initialization = 'template';
%ss_model = lpvssest(data_train, template_ss, options_pem_ss);
%[rms_est, rms_val] = rms_computation(data_train, data_test, ss_model, plot_)
