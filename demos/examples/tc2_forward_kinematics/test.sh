#! /bin/bash


for x in {1..30}
do
  if (($x % 10 == 0))
  then
    echo "Push FK repair log timer $x"
    git pull origin emsoft_examples_replayer_update1
    git add .
    git commit -m "Push FK repair log number $x"
    git push
  fi
  sleep 1
done

