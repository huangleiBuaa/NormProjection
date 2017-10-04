#!/bin/bash
methods=(PN)

lrs=(0.3)
Ts=(10 20 50 100)

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
   	th exp_MLP_MNIST.lua -model_method ${methods[$i]} -learningRate ${lrs[$j]}  -optimization simple -seed 1 -batchSize 256 -max_epoch 20 -T ${Ts[$k]}
      done
   done
done
