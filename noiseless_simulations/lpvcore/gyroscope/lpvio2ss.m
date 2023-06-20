function sys = lpvio2ss(A, B, na, nb, Ts, p_vec)
% LPV-SS realization (Shifted form, SISO, nb <= na)
    lpv_ss = [];
    A_SS = [];
    B_SS = [];
    for k = 1:na
        A_SS = [A_SS; -pshift(A.MatNoMonic{k}, k)];
    end
    for k = 1:nb
        B_SS = [B_SS; pshift(B.Mat{k + 1}, k)];
    end
    if na > nb
        B_SS = [B_SS; zeros(na - nb, 1)];
    end
    B_SS = B_SS + A_SS * B.Mat{1};
    A_SS = [A_SS, [eye(na - 1);  zeros(1, na - 1)]];
    C_SS = [1, zeros(1, na - 1)];
    D_SS = B.Mat{1};
    lpv_ss = lpvidss(A_SS, B_SS, C_SS, D_SS, 'innovation', zeros(na, 1),  ...
        [],  [], Ts);

    As = lpv_ss.A.matrices;
    Bs = lpv_ss.B.matrices;
    Cs = lpv_ss.C.matrices;
    Ds = lpv_ss.D.matrices;
    Ks = lpv_ss.K.matrices;
    A = As(:, :, 1);
    B = Bs(:, :, 1);
    for i = 2:size(As, 3)
        A = A + As(:, :, i) * p_vec{i - 1};
        B = B + Bs(:, :, i) * p_vec{i - 1};
    end
    C = Cs;
    D = Ds;
    K = Ks;
    sys = lpvidss(A, B, C, D, 'innovation', K, [], 1, 1);
end
