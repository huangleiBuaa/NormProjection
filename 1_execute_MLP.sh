#!/bin/bash
methods=(sgd Oblique PN_EI)

lrs=(1 0.3 0.1 0.03 0.01)
Ts=(1)

n=${#methods[@]}
m=${#lrs[@]}
f=${#Ts[@]}

for ((i=0;i<$n;++i))
do 
   for ((j=0;j<$m;++j))
   do	
     for ((k=0;k<$f;++k))
      do

    	echo "methods=${methods[$i]}"
    	echo "learningRates=${lrs[$j]}"
   	   echo "T=${Ts[$k]}"
   	th exp_MLP_MNIST.lua -model_method ${methods[$i]} -learningRate ${lrs[$j]}  -optimization simple -seed 1 -batchSize 256 -max_epoch 20
      done
   done
done
