#!/bin/bash

scp ./Dockerfile edu-gpu:~/temp/
rsync -avz --exclude 'outputs' --exclude 'datasets' --exclude '__pycache__' ./scripts/ edu-gpu:~/temp/

DOCKER_PS=$(ssh edu-gpu "sudo docker ps -q --filter ancestor=ljspeech-tensorflow")
for CONTAINER_ID in $DOCKER_PS; do
    ssh edu-gpu "sudo docker stop $CONTAINER_ID"
    ssh edu-gpu "sudo docker rm $CONTAINER_ID"
done

ssh edu-gpu "cd temp/ && ./build.sh"
DOCKER_ID=$(ssh edu-gpu "cd temp/ && sudo docker run -d --gpus all -v \$(pwd)/outputs:/opt/outputs ljspeech-tensorflow")
ssh edu-gpu "sudo docker logs -f $DOCKER_ID"
