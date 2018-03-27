
echo "Downloading AlexNet pretrained models..."

wget -c http://www.svcl.ucsd.edu/projects/cascade-rcnn/alexnet.zip

echo "Unzipping..."

unzip alexnet.zip && rm -f alexnet.zip

echo "Done."
