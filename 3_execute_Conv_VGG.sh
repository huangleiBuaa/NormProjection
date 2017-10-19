#!/bin/bash
#
#vggE_BN indicates the Normal architecture of Inception
#vggE_WN indicates the method of 'WN' in the paper
#vggE_PN indicates the method of 'NP-Epoch' in the paper
#vggE_PN_EI indicates the method of 'NP' in the paper
#vggE_Oblique_EI indicates the method of 'NP-Reim' in the paper
#
#
#

methods=(vggE_BN vggE_PN vggE_PN_EI vggE_Oblique_EI vggE_WN)
lrs=(0.1)
datasets=(./dataset/cifar10_original.t7)

batchSize=128
weightDecay=0.0005
dr=0
depth=28
widen_factor=1
nN=0
maxEpoch=160
eStep="{80,120}"


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
   CUDA_VISIBLE_DEVICES=1	th exp_Conv_CIFAR.lua -model ${methods[$i]} -learningRate ${lrs[$j]} -dataset ${datasets[$k]} -max_epoch ${maxEpoch} -seed 1 -dropout ${dr} -m_perGroup 64 -batchSize ${batchSize} -weightDecay ${weightDecay} -widen_factor ${widen_factor} -depth ${depth} -noNesterov ${nN} -epoch_step ${eStep}
      done
   done
done
