#!/bin/bash

scp ./Dockerfile edu-gpu:~/temp/
rsync -avz --exclude 'outputs' --exclude 'datasets' --exclude '__pycache__' ./scripts/ edu-gpu:~/temp/

ssh edu-gpu "cd temp/ && ./build.sh"
DOCKER_ID=$(ssh edu-gpu "sudo docker run -d --gpus all -v $PWD/outputs:/opt/outputs ljspeech-tensorflow")
ssh edu-gpu "sudo docker logs -f $DOCKER_ID"
