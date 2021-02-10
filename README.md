# Object Detection

if you are using cpu
```
docker build  -t ssdapp_mnozawa .
docker run -it -p 8501:8501 --name ssdapp_mnozawa -v $(pwd):/workspace ssdapp_mnozawa
```
if you are using gpu
```
docker build . -t ssdapp_mnozawa
docker run -it --gpus {your device} -p 8501:8501 --name ssdapp_mnozawa -v $(pwd):/workspace ssdapp_mnozawa
```

一応異なる画像サイズで学習させた画像を推論することができますが、かなり予測は悪いです