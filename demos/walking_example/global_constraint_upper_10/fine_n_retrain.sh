#!/bin/bash
# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# echo "remove the existing stats in:"
# echo $SCRIPT_DIR/tc1/finetuned_net/stats
# rm -rf $SCRIPT_DIR/tc1/finetuned_net/stats
# echo "Finetuning"
# for i in {1..50}
# do
#    echo "Fine-tuning - test: $i"
#    python3 retrain.py -it $i
# done

echo "repair multi models"
for i in {4..49}
do
   echo "Fine-tuning - test: $i"
   python3 net_repair_multi_models.py -id $i
done

# echo "remove the existing stats in:"
# echo $SCRIPT_DIR/tc1/retrained_net/stats
# rm -rf $SCRIPT_DIR/tc1/retrained_net/stats
# echo "Retraining"
# for i in {1..50}
# do
#    echo "Retrain - test: $i"
#    python3 net_retrain.py -vi 0 -sm 0 -sd 0 -vb 0
# done

# echo "Retraining"
# for i in {3..1}
# do
#    echo "Retrain - test: $i"
#    python net_repair.py -rl $i
# # done
# git add .
# git commit -m "update affine finetune"
# git push
