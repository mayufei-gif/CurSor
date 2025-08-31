%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% �ٶ���һ��HMM��ѵ�����HMM��
% ����һ��۲�ֵ�������HMM���������۲�ֵ��HMM��ƥ��ȡ�
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% O���۲�״̬��
O = 3;
% Q��HMM״̬��
Q = 4;
%ѵ�������ݼ�,ÿһ�����ݾ���һ��ѵ���Ĺ۲�ֵ
data=[1,2,3,1,2,2,1,2,3,1,2,3,1;
      1,2,3,1,2,2,1,2,3,1,2,3,1;
      1,2,3,1,2,2,1,2,3,1,2,3,1;
      1,2,3,1,2,2,1,2,3,1,2,3,1;
      1,2,3,1,2,2,1,2,3,1,2,3,1;
      1,2,3,1,2,2,1,2,3,1,2,3,1;
      1,2,3,1,2,2,1,2,3,1,2,3,1;
      1,2,3,1,2,2,1,2,3,1,2,3,1;
      1,2,3,1,2,2,1,2,3,1,2,3,1;
      1,2,3,1,2,2,1,2,3,1,2,3,1;]
% initial guess of parameters
% ��ʼ������
prior1 = normalise(rand(Q,1));
transmat1 = mk_stochastic(rand(Q,Q));
obsmat1 = mk_stochastic(rand(Q,O));
% improve guess of parameters using EM
% ��data���ݼ�ѵ�����������γ��µ�HMMģ��
[LL, prior2, transmat2, obsmat2] = dhmm_em(data, prior1, transmat1, obsmat1, 'max_iter', size(data,1));
% ѵ�������й۲�ֵ��HMMƥ���
LL
% ѵ����ĳ�ʼ���ʷֲ�
prior2
% ѵ�����״̬ת�Ƹ��ʾ���
transmat2
% �۲�ֵ���ʾ���
obsmat2
% use model to compute log likelihood
data1=[1,2,3,1,2,2,1,2,3,1,2,3,1]
loglik = dhmm_logprob(data1, prior2, transmat2, obsmat2)
% log lik is slightly different than LL(end), since it is computed after the final M step
% loglik ������data�����hmm(������Ϊprior2, transmat2, obsmat2)��ƥ��ֵ��Խ��˵��Խƥ�䣬0Ϊ����ֵ��
% pathΪviterbi�㷨�Ľ������������path
B = multinomial_prob(data1,obsmat2);
path = viterbi_path(prior2, transmat2, B)
save('sa.mat');