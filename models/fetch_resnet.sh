
echo "Downloading ResNet pretrained models..."

wget -c http://www.svcl.ucsd.edu/projects/cascade-rcnn/resnet.zip

echo "Unzipping..."

unzip resnet.zip && rm -f resnet.zip

echo "Done."
