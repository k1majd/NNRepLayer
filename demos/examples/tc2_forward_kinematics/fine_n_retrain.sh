#!/bin/bash
# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# echo "remove the existing stats in:"
# echo $SCRIPT_DIR/tc2/finetuned_net/stats
# rm -rf $SCRIPT_DIR/tc2/finetuned_net/stats
# echo "Finetuning"
# for i in {1..50}
# do
#    echo "Fine-tuning - test: $i"
#    python3 net_finetune.py -vi 0 -sm 0 -sd 0 -vb 0
# done

# echo "remove the existing stats in:"
# echo $SCRIPT_DIR/tc2/retrain_net/stats
# rm -rf $SCRIPT_DIR/tc2/retrain_net/stats
# echo "Retraining"
# for i in {1..50}
# do
#    echo "Retrain - test: $i"
#    python3 net_retrain.py -vi 0 -sm 0 -sd 0 -vb 0
# done

echo "Repair - finetune"
for i in {2..3}
do
   echo "repair_finetune - test: $i"
   python3 net_repair_finetune.py -rl $i -vi 0 -tl 86400 -mf 1
   git pull origin emsoft_examples_replayer_update1
   git add .
   git commit -m "update FK repair-finetune lay $i"
   git push
done
