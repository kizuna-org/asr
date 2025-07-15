URL ?= http://localhost:9115/probe?target=http://jenkins:8080&module=http_2xx

.PHONY: blackbox-status-check
blackbox-status-check:
	go run blackbox_status_check.go -url $(URL)
