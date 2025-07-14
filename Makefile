.PHONY: ansible
ansible:
	cd infra/ansible && ansible-playbook -i hosts.yml site.yml -C

.PHONY: ansible-mock
ansible-mock: up-mock
	cd infra/ansible && \
	ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook -i hosts.mock.yml site.yml -C

.PHONY: up-mock
up-mock: make-mock-key
	export PUB_KEY="$$(cat infra/mock/dind-host/dind-host.pub)" && \
	cd infra/mock && \
	docker compose build \
	  --build-arg PUB_KEY="$$PUB_KEY" && \
	docker compose up -d && \
	echo "root_password: $$ROOT_PASSWORD"

.PHONY: make-mock-key
make-mock-key: infra/mock/dind-host/dind-host infra/mock/dind-host/dind-host.pub

infra/mock/dind-host/dind-host & infra/mock/dind-host/dind-host.pub:
	cd infra/mock/dind-host && ssh-keygen -t rsa -b 4096 -f dind-host -N ""
	ssh-keygen -R '[localhost]:50022'
	chmod 600 infra/mock/dind-host/dind-host

.PHONY: clean
clean:
	rm infra/mock/dind-host/dind-host
	rm infra/mock/dind-host/dind-host.pub
