
echo "Downloading VggNet pretrained models..."

wget -c http://www.svcl.ucsd.edu/projects/cascade-rcnn/vggnet.zip

echo "Unzipping..."

unzip vggnet.zip && rm -f vggnet.zip

echo "Done."
