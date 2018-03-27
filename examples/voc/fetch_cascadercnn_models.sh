
echo "Downloading Cascade R-CNN model..."

wget -c http://www.svcl.ucsd.edu/projects/cascade-rcnn/voc-res101-9s-600-rfcn-cascade-pretrained.zip

echo "Unzipping..."

unzip voc-res101-9s-600-rfcn-cascade-pretrained.zip && rm -f voc-res101-9s-600-rfcn-cascade-pretrained.zip

echo "Done."
