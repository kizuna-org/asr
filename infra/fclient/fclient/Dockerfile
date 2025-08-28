FROM snowdreamtech/frpc:latest AS builder

FROM alpine:latest
COPY --from=builder /usr/bin/frpc /usr/bin/fclient
CMD ["/usr/bin/fclient", "-c", "/etc/fclient.toml"]
