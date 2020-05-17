function  M  = KISSme( f_d_neg_train,f_d_pos_train,R)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    B=f_d_neg_train*f_d_neg_train';
    A=f_d_pos_train*f_d_pos_train';
    A_tr=sum(sum(f_d_pos_train.*f_d_pos_train));
    B_tr=sum(sum(f_d_neg_train.*f_d_neg_train));
    n1 = size(f_d_pos_train,2);
    n2 = size(f_d_neg_train,2);
%    RR = rand(size(A))-0.5;
    M = inv(A/n1+R*eye(size(A)))-inv(B/n2+R*eye(size(B)));
%    Sx=eig(M);
    %M_tmp=M_tmp/mean(abs(Sx));
    %W=eye(size(A));
    %M=1*W+lambda*M_tmp;
end

