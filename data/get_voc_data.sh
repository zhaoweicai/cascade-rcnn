
echo "Downloading VOC data..."

wget -c http://www.svcl.ucsd.edu/projects/cascade-rcnn/voc_data.zip

echo "Unzipping..."

unzip voc_data.zip && rm -f voc_data.zip

echo "Done."
