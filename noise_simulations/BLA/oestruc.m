function [fits, best_order] = oestruc(training_data, test_data, orders)
    fits = zeros(size(orders, 1), 1);
    for i = 1:1:size(orders, 1)
        model = oe(training_data, orders(i, :));
        [yh, fit, x0] = compare(test_data, model, Inf);
        fits(i) = fit;
    end

    [max_val, max_ind] = max(fits);
    best_order = orders(max_ind, :);
end
