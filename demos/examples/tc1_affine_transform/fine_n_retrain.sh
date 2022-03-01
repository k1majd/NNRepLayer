#!/bin/bash
# echo "Finetuning"
# for i in {1..50}
# do
#    echo "Fine-tuning - test: $i"
#    python3 net_finetune.py -vi 0 -sm 0 -sd 0 -vb 0
# done

# echo "Retraining"
# for i in {1..50}
# do
#    echo "Retrain - test: $i"
#    python3 net_retrain.py -vi 0 -sm 0 -sd 0 -vb 0
# done

echo "Retraining"
for i in {3..1}
do
   echo "Retrain - test: $i"
   python net_repair.py -rl i
done
git add .
git commit -m "update affine repair"
git push
