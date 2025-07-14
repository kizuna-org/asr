.PHONY: ansible-playbook
ansible-playbook:
	cd infra/ansible && ansible-playbook -i hosts.yml site.yml
