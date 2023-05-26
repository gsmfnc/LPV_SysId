close all
fsize = 18;

cd('results')
wh_out
wh_real_out
cd('..')

bfr = compute_bfr(y, yr);
fig2 = figure(2);
plot(y); hold on;
plot(y - yr)
title("BFR=" + bfr + "%", 'fontsize', fsize)
legend('$y$', '$y-\hat y$', 'interpreter', 'latex', 'fontsize', fsize)

cd('results')
gyro_out
gyro_real_out
cd('..')

bfr = compute_bfr(y, yr);
fig1 = figure(1);
plot(y); hold on;
plot(y - yr)
title("BFR=" + bfr + "%", 'fontsize', fsize)
legend('$y$', '$y-\hat y$', 'interpreter', 'latex', 'fontsize', fsize)

function bfr = compute_bfr(y, yr)
num = 0;
den = 0;
mean_y = mean(yr);
for i = 1:1:size(y, 1)
    num = num + norm(y(i) - yr(i));
    den = den + norm(yr(i) - mean_y);
end

bfr = max(1 - num / den, 0) * 100
end
