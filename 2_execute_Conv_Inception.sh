#!/bin/bash
#
#googlenetbn indicates the Normal architecture of Inception
#googlenetbn_WN indicates the method of 'WN' in the paper
#googlenetbn_PN indicates the method of 'NP-Epoch' in the paper
#googlenetbn_PN_EI indicates the method of 'NP' in the paper
#googlenetbn_Oblique_EI indicates the method of 'NP-Reim' in the paper
#
#
#

methods=(googlenetbn googlenetbn_WN googlenetbn_PN googlenetbn_PN_EI googlenetbn_Oblique_EI)
lrs=(0.1)
datasets=(./dataset/cifar10_original.t7 ./dataset/cifar100_original.t7)

batchSize=64
weightDecay=0.0005
dr=0
depth=28
widen_factor=1
nN=0
maxEpoch=120
eStep="{50,80,100}"


n=${#methods[@]}
m=${#lrs[@]}
f=${#datasets[@]}

for ((i=0;i<$n;++i))
do 
   for ((j=0;j<$m;++j))
   do	
     for ((k=0;k<$f;++k))
      do

    	echo "methods=${methods[$i]}"
    	echo "learningRates=${lrs[$j]}"
   	echo "dataset=${datasets[$k]}"
   CUDA_VISIBLE_DEVICES=0	th exp_Conv_CIFAR.lua -model ${methods[$i]} -learningRate ${lrs[$j]} -dataset ${datasets[$k]} -max_epoch ${maxEpoch} -seed 1 -dropout ${dr} -m_perGroup 64 -batchSize ${batchSize} -weightDecay ${weightDecay} -widen_factor ${widen_factor} -depth ${depth} -noNesterov ${nN} -epoch_step ${eStep}
      done
   done
done
