#! /bin/bash


for x in {1..172800}
do
  if (($x % 3600 == 0))
  then
    echo "Push FK repair log timer $x"
    git pull origin emsoft_examples_replayer_update1
    git add .
    git commit -m "Push FK repair log number $x"
    git push
  fi
  sleep 1
done

