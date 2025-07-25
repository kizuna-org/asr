.PHONY: init
init:
	cd infra/ansible && ansible-galaxy install -r requirements.yml

.PHONY: ansible
ansible: init
	cd infra/ansible && ansible-playbook -i hosts.yml site.yml -C

.PHONY: ansible-mock
ansible-mock: init mock-up
	cd infra/ansible && \
	ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook -i hosts.mock.yml site.yml -C

.PHONY: ansible-mock-apply
ansible-mock-apply: init mock-up
	cd infra/ansible && \
	ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook -i hosts.mock.yml site.yml

.PHONY: mock-test
mock-test: init ansible-mock-apply mock-up
	# cd infra/mock && \
	# docker compose 
	# cd infra/test && \
	# docker build -t blackbox-status-check . && \
	# docker run --rm -e BLACKBOX_URL=http://localhost:9115/probe?target=http://jenkins:8080&module=http_2xx blackbox-status-check

.PHONY: mock-up
mock-up: init make-mock-key
	export PUB_KEY="$$(cat infra/mock/dind-host/dind-host.pub)" && \
	cd infra/mock && \
	cd dind-host && \
	DOCKER_BUILDKIT=1 docker compose build \
	  --build-arg PUB_KEY="$$PUB_KEY" && \
	docker compose up -d

.PHONY: mock-down
mock-down: init
	cd infra/mock && docker compose down

.PHONY: make-mock-key
make-mock-key: infra/mock/.mock-key.stamp

infra/mock/.mock-key.stamp: \
	infra/mock/dind-host/dind-host \
	infra/mock/dind-host/dind-host.pub \
	infra/mock/etc/letsencrypt/live/frps-connect.shiron.dev/fullchain.pem \
	infra/mock/etc/letsencrypt/live/frps-connect.shiron.dev/privkey.pem
	touch infra/mock/.mock-key.stamp

infra/mock/dind-host/dind-host infra/mock/dind-host/dind-host.pub:
	cd infra/mock/dind-host && \
	ssh-keygen -t rsa -b 4096 -f dind-host -N ""
	ssh-keygen -R '[localhost]:50022'
	chmod 600 infra/mock/dind-host/dind-host

infra/mock/etc/letsencrypt/live/frps-connect.shiron.dev/fullchain.pem \
infra/mock/etc/letsencrypt/live/frps-connect.shiron.dev/privkey.pem:
	mkdir -p infra/mock/etc/letsencrypt/live/frps-connect.shiron.dev
	scp snct-proxy-srv.shiron.dev:~/frps-connect.shiron.dev/fullchain.pem infra/mock/etc/letsencrypt/live/frps-connect.shiron.dev/fullchain.pem
	scp snct-proxy-srv.shiron.dev:~/frps-connect.shiron.dev/privkey.pem infra/mock/etc/letsencrypt/live/frps-connect.shiron.dev/privkey.pem

.PHONY: clean
clean:
	cd infra/mock && docker compose down
	rm -f infra/mock/dind-host/dind-host
	rm -f infra/mock/dind-host/dind-host.pub
	rm -f infra/mock/etc/letsencrypt/live/frps-connect.shiron.dev/fullchain.pem
	rm -f infra/mock/etc/letsencrypt/live/frps-connect.shiron.dev/privkey.pem
	rm -f infra/mock/.mock-key.stamp
	ssh-keygen -R '[localhost]:50022'
