# prepare dataset
echo prepare dataset...
cd src/input
curl http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O
tar -xvf VOCtrainval_11-May-2012.tar
rm VOCtrainval_11-May-2012.tar

# model weight setting
echo weight setting...
cd ../experiments/_trained
mkdir weight
cd weight
curl https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth -O
mv ssd300_mAP_77.43_v2.pth seed_0.pt

echo complete!