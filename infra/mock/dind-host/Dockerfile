FROM docker:dind

ARG PUB_KEY

RUN apk add --no-cache openssh sudo
RUN ssh-keygen -A
RUN adduser -D -h /home/ansible_user ansible_user
RUN passwd -u ansible_user
RUN chown ansible_user:ansible_user /etc/ssh/ssh_host_*_key

RUN if [ -z "${PUB_KEY}" ]; then echo "Error: PUB_KEY build-arg is not set"; exit 1; fi && \
  mkdir -p /home/ansible_user/.ssh && \
  echo "${PUB_KEY}" > /home/ansible_user/.ssh/authorized_keys

RUN chown -R ansible_user:ansible_user /home/ansible_user && \
  chmod 700 /home/ansible_user/.ssh && \
  chmod 600 /home/ansible_user/.ssh/authorized_keys

RUN echo "ansible_user ALL=(ALL) NOPASSWD: /usr/bin/docker" > /etc/sudoers.d/ansible-docker && \
  chmod 0440 /etc/sudoers.d/ansible-docker

RUN sed -i 's/^#?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config && \
  sed -i 's/^#?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
  echo 'PubkeyAuthentication yes' >> /etc/ssh/sshd_config

EXPOSE 22

RUN apk add --no-cache \
  docker-compose \
  git \
  bash \
  jq \
  python3

CMD ["sh", "-c", "/usr/sbin/sshd && exec dockerd-entrypoint.sh dockerd"]
