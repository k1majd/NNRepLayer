poetry shell
cd $2
echo "Path to folder is:\t$2"
echo "Setting up Gurobi for python"
sudo $1 setup.py install
exit