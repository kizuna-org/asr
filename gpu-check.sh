rsync -avz ./gpu-check/ edu-gpu:~/temp/gpu-check/
ssh edu-gpu "cd ~/temp/gpu-check/ && chmod +x build.sh && ./build.sh"
