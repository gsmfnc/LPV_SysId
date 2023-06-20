close all
fsize = 18;

cd('results')
gyro_out
gyro_real_out
cd('..')

bfr = compute_bfr(y, yr);
fig1 = figure(1);
plot(yr); hold on;
plot(y - yr)
title("BFR=" + bfr + "%", 'fontsize', fsize)
legend('$y$', '$y-\hat y$', 'interpreter', 'latex', 'fontsize', fsize)

function bfr = compute_bfr(y, yr)
num = 0;
den = 0;
mean_y = mean(yr);
num = norm(y - yr);
den = norm(yr - mean_y);

bfr = max(1 - num / den, 0) * 100
end
