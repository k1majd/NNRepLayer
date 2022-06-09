#!/bin/bash


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
# done
python3 $SCRIPT_DIR/net_repair.py
git add .
git commit -m "tc7 dynamic constraint layer 3"
git push
