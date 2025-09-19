# --- builder stage ---
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY blackbox_status_check.go .
RUN go build -o blackbox_status_check blackbox_status_check.go

# --- runtime stage ---
FROM alpine:3.20
WORKDIR /app
COPY --from=builder /app/blackbox_status_check .

# URLは環境変数BLACKBOX_URLで指定
ENV BLACKBOX_URL=""
ENTRYPOINT ["/app/blackbox_status_check", "--url", "${BLACKBOX_URL}"]
