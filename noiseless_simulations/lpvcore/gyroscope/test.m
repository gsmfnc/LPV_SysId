%% Data presettings
clearvars; close all; clc;
addpath('../data/')

%% Settings:
value_for_ZeroIsNonFree = false;

%% Load data
load estim.mat
exp_y = y_q4d;
exp_p = [cos(q_all(:,2)),qd_all(:,1),sin(q_all(:,2))];
exp_u = u_i2;
data_LPV_est=lpviddata(exp_y,exp_p,exp_u,Ts);

load valid.mat;
exp_y = y_q4d;
exp_p = [cos(q_all(:,2)),qd_all(:,1),sin(q_all(:,2))];
exp_u = u_i2;
data_LPV_val=lpviddata(exp_y,exp_p,exp_u,Ts);

t10 = (0:Ts:(Ts*(1E4-1)))';
t30 = (0:Ts:(Ts*(3E4-1)))';

%% Plot data-sets
figure;
tiles = tiledlayout(4,2);
% ################
nexttile; box on;
plot(t10,data_LPV_est.InputData)
title('Estimation data')
ylabel('$i_2$ (A)')
nexttile;box on;
plot(t30,data_LPV_val.InputData)
title('Validation data')
% ################
nexttile;box on;
plot(t10,data_LPV_est.SchedulingData(:,2))
ylabel('$\dot{q}_1$ (rad/s)'); 
nexttile;box on;
plot(t30,data_LPV_val.SchedulingData(:,2)); 
% ################
nexttile; hold on; box on;
plot(t10,data_LPV_est.SchedulingData(:,3), 'DisplayName','$\sin(q_2)$')
plot(t10,data_LPV_est.SchedulingData(:,1), 'DisplayName','$\cos(q_2)$')
legend('show');
nexttile; hold on; box on;
plot(t30,data_LPV_val.SchedulingData(:,3), 'DisplayName','$\sin(q_2)$')
plot(t30,data_LPV_val.SchedulingData(:,1), 'DisplayName','$\cos(q_2)$')
legend('show');
% ################
nexttile;box on;
plot(t10,data_LPV_est.OutputData)
ylabel('$\dot{q}_4$ (rad/s)')
xlabel('Time (s)')
nexttile;box on;
plot(t30,data_LPV_val.OutputData)
xlabel('Time (s)')
set(tiles,'TileSpacing','tight', 'Padding','tight')


%% Black box LPV-IO ID
% Scheduling and configuration
s2 = preal('sin(q2)', 'dt');
c2 = preal('cos(q2)', 'dt');
q1d = preal('q1d', 'dt');

% LPV-ARX estimation
A = randn(1) + randn(1)*q1d + randn(1)*s2 + randn(1)*c2;
B = randn(1) + randn(1)*q1d + randn(1)*s2 + randn(1)*c2;
na = 5;
nb = 5;
A_poly = eye(1);
for k=1:na
    if k==1
        A_poly={A_poly, pshift(A,-k)};
    else
        A_poly={A_poly{:,1:k}, pshift(A,-k)};
    end
end
B_poly=rand(1);
for k=1:nb
    if k==1
        B_poly={B_poly, pshift(B,-k)};
    else
        B_poly={B_poly{:,1:k}, pshift(B,-k)};
    end
end
template_arx = lpvidpoly(A_poly, B_poly, [], [], [], 0, Ts, ...
    'ZeroIsNonFree', value_for_ZeroIsNonFree);
template_arx.InputName = {'Current Gimbal 2'};
template_arx.InputUnit = {'A'};
template_arx.OutputName = {'q4d'};
template_arx.OutputUnit = {'rad/s'};
disp(template_arx);
m_LPV_arx = lpvarx(data_LPV_est, template_arx, options);
m_LPV_arx_ss=lpvio2ss(m_LPV_arx.A,m_LPV_arx.B,na,nb,Ts);

% LPV-OE estimation
template_oe = lpvidpoly([], m_LPV_arx.B, [], [], m_LPV_arx.A, 0, Ts, ...
    'ZeroIsNonFree',value_for_ZeroIsNonFree);
template_oe.InputName = {'Current Gimbal 2'};
template_oe.InputUnit = {'A'};
template_oe.OutputName = {'q4d'};
template_oe.OutputUnit = {'rad/s'};
options_oe=lpvoeOptions('Display', 'off', ...
    'SearchOptions',  struct('StepSize', 10, 'StepTolerance', 1E-10), ...
    'Regularization', struct('Lambda', 1),...
    'Initialization', 'template');
m_LPV_oe = lpvoe(data_LPV_est, template_oe, options_oe);
m_LPV_oe_ss=lpvio2ss(m_LPV_oe.F, m_LPV_oe.B, na, nb, Ts);


% LPV PEM-SS estimation
template_ss=m_LPV_oe_ss;  % oe-based initialisation
options_pem_ss = lpvssestOptions;
options_pem_ss.Display='off';
m_LPV_ss_pem= lpvssest(data_LPV_est, template_ss, options_pem_ss);

%% Comparision of LPVID results
figure('Name', 'Comparison ARX OE & SS-PEM - 1 step');
compare(data_LPV_val,m_LPV_arx,m_LPV_oe,m_LPV_ss_pem,1); hold on
[y_1step, ~]=compare(data_LPV_val,m_LPV_arx,m_LPV_oe,m_LPV_ss_pem,1);
plot(t30,y_1step{1}-data_LPV_val.OutputData); hold on
plot(t30,y_1step{2}-data_LPV_val.OutputData); hold on
plot(t30,y_1step{3}-data_LPV_val.OutputData); hold on

figure('Name','Comparison ARX OE & SS-PEM - Inf step');
compare(data_LPV_val,m_LPV_arx,m_LPV_oe,m_LPV_ss_pem,Inf); hold on
[y_Infstep, ~]=compare(data_LPV_val,m_LPV_arx,m_LPV_oe,m_LPV_ss_pem,Inf);
plot(t30,y_Infstep{1}-data_LPV_val.OutputData); hold on
plot(t30,y_Infstep{2}-data_LPV_val.OutputData); hold on
plot(t30,y_Infstep{3}-data_LPV_val.OutputData); hold on



%%%%%%%%%%%%%%%%%%%%%%%   LOCAL FUNCTIONS   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function lpv_ss=lpvio2ss(A,B,na,nb,Ts)
% LPV-SS realsiation (Shifted form, SISO, nb<=na)
lpv_ss=[];
A_SS=[];
B_SS=[];
for k=1:na
    A_SS=[A_SS;-pshift(A.MatNoMonic{k},k)];
end
for k=1:nb
    B_SS=[B_SS;pshift(B.Mat{k+1},k)];
end
if na>nb
    B_SS=[B_SS;zeros(na-nb,1)];
end
B_SS=B_SS+A_SS*B.Mat{1};
A_SS=[A_SS,[eye(na-1); zeros(1,na-1)]];
C_SS=[1,zeros(1,na-1)];
D_SS=B.Mat{1};
lpv_ss=lpvidss(A_SS,B_SS,C_SS,D_SS,'innovation',zeros(na,1), [], [],Ts);
end

