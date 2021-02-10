docker build  -t ssdapp_mnozawa .
docker run -it -p 8501:8501 --name ssdapp_mnozawa -v $(pwd):/workspace ssdapp_mnozawa
