rsync -avz ./poc/ edu-gpu:~/temp/poc/
ssh edu-gpu "cd ~/temp/poc && sudo docker compose down && sudo docker compose up --build"
