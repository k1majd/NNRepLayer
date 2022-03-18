@ECHO OFF 
ECHO running adv sampler
python acas_adv_sampler.py
git pull origin emsoft_examples_replayer_update1
git add .
git commit -m "update acas adv samples"
git push