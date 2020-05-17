%修改opt更改实验设置
opt.protocol = 'small';%large or small
opt.feat = 'CNN';%CNN or FV
opt.Dim = 40;
opt.cascade=4 ;
opt.breakpoint=cell(1,opt.cascade);
opt.metric = 'XQDA'; %RMML or XQDA or KISSME
interval = opt.Dim/2^(opt.cascade-1);
for i = 1:opt.cascade
    tmp = 0;
    for j = 1:2^(opt.cascade-i)+1
        opt.breakpoint{i} = [opt.breakpoint{i},tmp];
        tmp = tmp + interval;
 %%%%%%%%%%%%%%是否ensemble%%%%%%%%%%%%%%%%%
 %       opt.breakpoint{i} = [0,opt.Dim ];
 %%%%%%%%%%%%%%是否ensemble%%%%%%%%%%%%%%%%%
    end
    interval = 2 * interval;
end
opt.T=1e-10;
opt.beta=0.1*ones(1,opt.cascade);
%opt.beta=[0.2,0.2,0.1];
opt.splitrecord=cell(1,opt.cascade);
opt.Lrecord=cell(1,opt.cascade);
opt.sample_N=40;
if strcmp(opt.feat,'CNN')
    if ~exist('f','var')
        f = importdata('f.mat');
    end
    if strcmp(opt.protocol,'large')
        train_sample = 450000;
        test_start = 450000;
        test_end = 500000;
        test_shift =30;
    elseif strcmp(opt.protocol,'small')
        train_sample = 6000;
        test_start = 6000;
        test_end = 10000;
        test_shift =40;      
    else
        ME = MException('MyComponent:noSuchVariable','undefined protocol %s', opt.protocol);
        throw(ME);
    end
elseif strcmp(opt.feat,'FV')
    if ~exist('f','var')
        f = importdata('encoding.mat');
    end
    train_sample = 100000;
    test_start = 100000;
    test_end = 110000;
    test_shift = 40;
else
    ME = MException('MyComponent:noSuchVariable','undefined feature %s', opt.feat);
    throw(ME);
end

label=importdata('pairwise_label.txt');
%%%%%%%%%%%%%%%%%%%%%%%%%%pairwise label%%%%%%%%%%%%%%%%%%%%%%
label_train=label(1:train_sample);
label_a=[label_train(opt.sample_N:train_sample);label_train(1:opt.sample_N-1)];
label_pair_train=(label_train==label_a);
label_pair_train=single(label_pair_train);
labels_train=(1-label_pair_train);
labels_train(labels_train==0)=-1;

label_test=label(test_start+1:test_end);
label_a_test=[label_test(test_shift+1:end);label_train(1:test_shift)];
label_pair_test=(label_test==label_a_test);
label_pair_test=single(label_pair_test);
labels_test=(1-label_pair_test);
labels_test(labels_test==0)=-1;
%%%%%%%%%%%%%%%%%%%%%%%%%%pairwise label%%%%%%%%%%%%%%%%%%%%%%

run vlfeat-0.9.20\toolbox\vl_setup.m
meanvalue=mean(f(:,1:train_sample),2);
if ~exist('s1','var')
    [s1,s2,s3]=pca(f(:,1:train_sample)');
end
%%%%%%%%%%%%%%%测试次数%%%%%%%%%%%%%%%
%f_or = s1(:,1:opt.Dim)'*bsxfun(@minus,f,meanvalue);
%tmp_label_record=[];
EER_record_train=[];
EER_record_test=[];
for iter = 1:1
%%%%%%%%%%%%%%%测试次数%%%%%%%%%%%%%%%
f_=s1(:,1:opt.Dim)'*bsxfun(@minus,f,meanvalue);
%f_=f;
tic
for j=1:opt.cascade
    f_1=[];
    f_2=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%是否shuffule%%%%%%%%%%%%%%%%%%%%
    R=randperm(size(f_,1));
    f_R=f_(R,:);
    f_=f_R;
    opt.splitrecord{j}=R;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%是否shuffule%%%%%%%%%%%%%%%%%%%% 
    for i=1:length(opt.breakpoint{j})-1
        f_tmp=f_(opt.breakpoint{j}(i)+1:opt.breakpoint{j}(i+1),:);
        f_train=f_tmp(:,1:train_sample);
        
        a=[f_train(:,opt.sample_N:train_sample),f_train(:,1:opt.sample_N-1)];
        f_d_train=f_train-a;


        f_d_pos_train=f_d_train(:,label_pair_train==1);
        f_d_neg_train=f_d_train(:,label_pair_train==0);

    
        n1=length(nonzeros(label_pair_train==1));
        n2=length(nonzeros(label_pair_train==0));
        disp(['n1=',num2str(n1)]);
        disp(['n2=',num2str(n2)]);


        if strcmp(opt.metric,'RMML')
            M2  = RMML( f_d_neg_train,f_d_pos_train,opt.beta(j));   
        elseif strcmp(opt.metric,'KISSME')
            M2  = KISSme( f_d_neg_train,f_d_pos_train,0);
        elseif strcmp(opt.metric,'XQDA')
            [W, M,inCov, exCov] = XQDA(f_train', [f_train(:,opt.sample_N:train_sample),f_train(:,1:opt.sample_N-1)]', label_train, [label_train(opt.sample_N:train_sample);label_train(1:opt.sample_N-1)]);
            M2 = W*M*W';
        else
            ME = MException('MyComponent:noSuchVariable','undefined metric %s', opt.metric);
            throw(ME);
        end
        %print training EER
        p_train=sum(f_d_train.*f_d_train);
        [~, ~, info1] = vl_roc(labels_train, p_train) ;
        b=(M2)*f_d_train;

        p_train=sum(f_d_train.*b);
        [~, ~, info2] = vl_roc(labels_train, p_train) ;
        record2{1}=info2;
        disp(['eer1=',num2str(info1.eer)]);
        disp(['eer2=',num2str(info2.eer)]);
        %test
        
        f_test=f_tmp(:,test_start+1:test_end);
        a=[f_test(:,test_shift+1:end),f_test(:,1:test_shift)];
        f_d_test= f_test-a;

        p_test=sum(f_d_test.*f_d_test);


        [~, ~, info3] = vl_roc(labels_test, p_test) ;
        b=(M2)*f_d_test;
        p_test=sum(f_d_test.*b);
        [~, ~, info4] = vl_roc(labels_test, p_test) ;
        disp(['eer3=',num2str(info3.eer)]);
        disp(['eer4=',num2str(info4.eer)]);
        disp(['test n1=',num2str(length(nonzeros(labels_test==-1)))]);
        disp(['test n2=',num2str(length(nonzeros(labels_test==1)))]);
        
        
        %test
        [p1,p2]=eig(M2);
        p2_diag = diag(p2);
        p2_diag(p2_diag<=opt.T)=opt.T;
        p2 = diag(p2_diag);
        L=p1*sqrt(p2);
        f_1=L'*f_tmp;
        opt.Lrecord{j}(:,:,i)=L;

        f_1=sign(f_1).*((abs(f_1)).^(1/2));  
        f_2=[f_2;f_1];
    end
    f_=f_2;
    disp(['layer ',num2str(j),' end.....'])
    disp(['-------------------------------------------']);

end
%%%%%%%%%%%%%%%%%%%%%%%%%
    EER_record_train = [EER_record_train,info2.eer];
    EER_record_test = [EER_record_test,info4.eer];
end
disp(['training EER is ', num2str(mean(EER_record_train))])
disp(['testing EER is ', num2str(mean(EER_record_test))])
%%%%%%%%%%%%%%%%%%%%%%%%%

toc
