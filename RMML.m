function  M  = RMML( f_d_neg_train,f_d_pos_train,lambda)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    B=f_d_neg_train*f_d_neg_train';
    A=f_d_pos_train*f_d_pos_train';
    A_tr=sum(sum(f_d_pos_train.*f_d_pos_train));
    B_tr=sum(sum(f_d_neg_train.*f_d_neg_train));
    M_tmp=(B/B_tr-1*A/A_tr);
    Sx=eig(M_tmp);
    M_tmp=M_tmp/mean(abs(Sx));
    W=eye(size(A));
    M=1*W+lambda*M_tmp;
end

