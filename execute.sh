
#!/bin/sh
python examples/tc4_robot_control/net_train.py -ep 100 -nt 100
git add .
git commit -m "test update3"
git push