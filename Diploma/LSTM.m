function net1=LSTM(input_size , before_layers , before_activation,hidden_size, after_layers , after_activations , output_size)
%% this part split the input into two seperate parts the first part
%is the input size and the second part is the memory
real_input_size=input_size ;
N_before=length(before_layers);
N_after=length(after_layers) ;
delays_vec=1 ;
if (N_before>0 ) && (N_after>0)
input_size=before_layers(end) ;
net1=fitnet( [before_layers , input_size+hidden_size , hidden_size*ones(1,9),after_layers]) ;
elseif (N_before>0) && (N_after==0)
input_size=before_layers(end) ;
net1=fitnet([before_layers,input_size+hidden_size , hidden_size*ones(1 , 9)]) ;
elseif (N_before==0)&&(N_after>0)
net1=fitnet([input_size+hidden_ size , hidden_size*ones(1, 9) , after_layers]) ;
else
net1 =fitnet( [input size+hidden_size, hidden_size*ones(1, 9)]);
end
net1=configure(net1 ,rand( real_input_size , 200) , rand(output_size,200)) ;
%% concatenation
net1.layers{N_before+1}.name='Concatenation Layer';
net1.layers{N_before+2}.name = 'Forget Amount' ;
net1.layers{N_before+3}.name= 'Forget Gate';
net1.layers{N_before+4}.name= 'Remember Amount';
net1.layers{N_before+5}.name= 'tanh Input' ;
net1.layers{N_before+6}.name= 'Forget Gate';
net1.layers{N_before+7}.name= 'Update Memory';
net1.layers {N_before+8}.name= 'tanh Memory';
net1.layers{N_before+9}.name= 'Combine Amount' ;
net1.layers{N_before+10}.name= 'Combine gate' ;
net1.layerConnect(N_before+3 , N_before+7) =1 ;
net1.layerConnect(N_before+1 ,N_before+10)=1 ;
net1.layerConnect(N_before+4 , N_before+3)=0;
net1.layerWeights{N_before+1 , N_before+10}.delays=delays_vec ;
if N_before>0
net1.LW{N_before+1 , N_before} = [eye(input_size) ; zeros(hidden_size, input_size)];
else
net1.IW{1,1}=[eye( input_size) ;zeros(hidden_size , input_size)];
end
net1.LW{N_before+1 , N_before+10}=repmat ([zeros(input_size, hidden_size); eye(hidden_size)] , [1 , size(delays_vec,2)] ) ;
net1.layers{N_before+1}.transferFcn='purelin';
net1.layerWeights{N_before+1 ,N_before+10}.learn=false;
if N_before>0
net1.layerWeights{ N_before+1 ,N_before}.learn=false;
else
net1.inputWeights{ 1, 1}.learn=false ;
end
%%
net1.biasConnect = [ones(1,N_before) 0 1 0 1 1 0 0 0 1 0 1 ones(1,N_after)]' ;%
%% first gate
net1.layers{N_before+2}.transferFcn= 'logsig' ;
net1.layerWeights{N_before+3, N_before+2}.weightFcn='scalprod' ;
% net1 .layerWeights{3 , 7} .weightFcn= ' scalprod ';
net1.layerWeights{N_before+3, N_before+2}.learn=false;
net1.layerWeights{N_before+3, N_before+7}.learn=false ;
net1.layers{N_before+3}.netinputFcn= 'netprod';
net1.layers{N_before+3}.transferFcn='purelin';
net1.LW{N_before+3, N_before+2}=1;
% net1.LW{3 , 7} =1 ;
%% second gate
net1.layerConnect(N_before+4,N_before+1)=1;
net1.layers{N_before+4}.transferFcn='logsig' ;
%% tanh
net1.layerConnect(N_before+5 , N_before+4) =0;
net1.layerConnect( N_before+5 , N_before+1)=1;
%%second gate mult
net1.layerConnect(N_before+6, N_before+4)=1;
net1.layers{N_before+6}.netinputFcn='netprod' ;
net1.layers{N_before+6} .transferFcn= 'purelin';
net1.layerWeights{N_before+6, N_before+5}.weightFcn='scalprod';
net1.layerWeights {N_before+6 , N_before+4}.weightFcn='scalprod';
net1.layerWeights{N_before+6 , N_before+5}.learn=false ;
net1.layerWeights{N_before+6,N_before+4}.learn=false;
net1.LW{N_before+6 , N_before+5} =1;
net1.LW{N_before+6 , N_before+4}=1 ;
%% C update
delays_vec=1;
net1.layerConnect(N_before+7,N_before+3)=1 ;
net1.layerWeights{N_before+3,N_before+7} . delays=delays_vec ;
net1.layerWeights{N_before+7,N_before+3}.weightFcn= 'scalprod';
net1.layerWeights{N_before+7,N_before+6}.weightFcn= 'scalprod';
net1 .layers{N_before+7}.transferFcn= 'purelin';
net1.LW{N_before+7 , N_before+3} =1 ;
net1.LW{N_before+7 , N_before+6} =1 ;
net1.LW{N_before+3 , N_before+7}=repmat(eye(hidden_size), [1 , size(delays_vec,2)] );
net1.layerWeights{N_before+3 , N_before+7}.learn=false ;
net1.layerWeights{N_before+7 ,N_before+6}.learn=false;
net1.layerWeights{N_before+7,N_before+3}.learn=false;
%% output stage
net1.layerConnect(N_before+9, N_before+8)=0;
net1.layerConnect(N_before+10 , N_before+8) = 1 ;
net1.layerConnect(N_before+9, N_before+1) =1 ;
net1.layerWeights{N_before+10 , N_before+8}.weightFcn='scalprod' ;
net1.layerWeights{N_before+10 , N_before+9}.weightFcn= 'scalprod' ;
net1.LW{N_before +10 ,N_before+9}=1 ;
net1.LW{N_before+10,N_before+8}=1 ;
net1.layers{N_before+10}.netinputFcn= 'netprod' ;
net1.layers{N_before+10}.transferFcn= 'purelin';
net1.layers{N_before+9}.transferFcn= 'logsig';
net1.layers{N_before+5}.transferFcn='tansig'; 
net1.layers{N_before+8}.transferFcn='tansig' ;
net1.layerWeights{N_before+10 ,N_before+ 9}.learn= false ;
net1.layerWeights{N_before +10,N_before+8 }.learn= false ;
net1.layerWeights{N_before+7 ,N_before+3 }. learn=false ;
for ll=1:N_before
net1.layers{ll}.transferFcn=before_activation;
end
for ll=1:N_after
net1. layers{end-ll}.transferFcn=after_activations ;
end

net1.layerWeights{N_before+8 , N_before+7}.weightFcn='scalprod' ;
net1.LW{N_before+8 , N_before+7}=1 ;
net1.layerWeights{N_before+8 , N_before+7}.learn=false ;
%%
net1=configure(net1 , rand(real_input_size ,200) , rand(output_size , 200) ) ;
net1.trainFcn= 'trainlm';
end