clc
clear
close all

training_ = false;

[training_data, test_data, Ts] = load_dataset();

if training_ == true
    orders = struc(1:10, 1:10, 1:10);
    [fits, best_order] = oestruc(training_data, test_data, orders);
else
    load model
    load fits
    load best_order
end

model = oe(training_data, best_order);
compare(test_data, model, Inf)
