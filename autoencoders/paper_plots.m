close all
fsize = 18;

cd('res')
gyro_output
gyro_real_output
cd('..')

bfr = compute_bfr(y, yr);
fig1 = figure(1);
plot(yr); hold on;
plot(y - yr)
title("BFR=" + bfr + "%", 'fontsize', fsize)
legend('$y$', '$y-\hat y$', 'interpreter', 'latex', 'fontsize', fsize)

cd('res')
wh_output_20
wh_real_output_20
cd('..')

bfr = compute_bfr(y, yr);
fig2 = figure(2);
plot(yr); hold on;
plot(y - yr)
title("BFR=" + bfr + "%", 'fontsize', fsize)
legend('$y$', '$y-\hat y$', 'interpreter', 'latex', 'fontsize', fsize)

cd('res')
wh_output_10
wh_real_output_10
cd('..')

bfr = compute_bfr(y, yr);
fig3 = figure(3);
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
