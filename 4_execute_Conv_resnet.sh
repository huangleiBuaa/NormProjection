#!/bin/bash
#
#old_r_BN indicates the Normal architecture of residual network
#old_r_WN indicates the method of 'WN' in the paper
#old_r_PN indicates the method of 'NP-Epoch' in the paper
#old_r_PN_EI indicates the method of 'NP' in the paper
#old_r_Oblique_EI indicates the method of 'NP-Reim' in the paper
#
#
#

methods=(old_r_BN old_r_WN old_r_PN old_r_PN_EI old_r_Oblique_EI)
depths=(20 32 44 56 110)
datasets=(./dataset/cifar10_original.t7 ./dataset/cifar100_original.t7)

batchSize=128
weightDecay=0.0001
dr=0
widen_factor=1
nN=0
maxEpoch=160
eStep="{80,120}"
learningRateDecayRatio=0.1


n=${#methods[@]}
m=${#depths[@]}
f=${#datasets[@]}

for ((i=0;i<$n;++i))
do 
   for ((j=0;j<$m;++j))
   do	
     for ((k=0;k<$f;++k))
      do

    	echo "methods=${methods[$i]}"
    	echo "depths=${depths[$j]}"
   	echo "dataset=${datasets[$k]}"
   CUDA_VISIBLE_DEVICES=0	th exp_Conv_CIFAR.lua -model ${methods[$i]} -learningRate 0.1 -depth ${depths[$j]} -dataset ${datasets[$k]} -max_epoch ${maxEpoch} -seed 1 -dropout ${dr} -m_perGroup 64 -batchSize ${batchSize} -weightDecay ${weightDecay} -widen_factor ${widen_factor}  -noNesterov ${nN} -epoch_step ${eStep} -learningRateDecayRatio ${learningRateDecayRatio}
      done
   done
done
