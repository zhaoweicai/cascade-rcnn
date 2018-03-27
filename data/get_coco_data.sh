
echo "Downloading COCO data..."

wget -c http://www.svcl.ucsd.edu/projects/cascade-rcnn/coco_data.zip

echo "Unzipping..."

unzip coco_data.zip && rm -f coco_data.zip

echo "Done."
