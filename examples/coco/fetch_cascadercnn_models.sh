
echo "Downloading Cascade R-CNN model..."

wget -c http://www.svcl.ucsd.edu/projects/cascade-rcnn/coco-res101-15s-800-fpn-cascade-pretrained.zip

echo "Unzipping..."

unzip coco-res101-15s-800-fpn-cascade-pretrained.zip && rm -f coco-res101-15s-800-fpn-cascade-pretrained.zip

wget -c http://www.svcl.ucsd.edu/projects/cascade-rcnn/coco-res50-15s-800-fpn-cascade-pretrained.zip

echo "Unzipping..."

unzip coco-res50-15s-800-fpn-cascade-pretrained.zip && rm -f coco-res50-15s-800-fpn-cascade-pretrained.zip

wget -c http://www.svcl.ucsd.edu/projects/cascade-rcnn/coco-res50-15s-800-fpn-base-pretrained.zip

echo "Unzipping..."

unzip coco-res50-15s-800-fpn-base-pretrained.zip && rm -f coco-res50-15s-800-fpn-base-pretrained.zip

echo "Done."
