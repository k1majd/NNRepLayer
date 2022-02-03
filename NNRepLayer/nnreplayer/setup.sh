echo "enter the complete file path setup.py file"
read filePath
poetry shell
cd $filePath
ls
python3 setup.py install
exit