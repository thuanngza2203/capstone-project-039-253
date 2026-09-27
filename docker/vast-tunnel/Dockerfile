# SSH tunnel tới vLLM trên Vast.ai (thay cho terminal chạy lệnh ssh -L trên Windows).
FROM alpine:3.20
RUN apk add --no-cache openssh-client
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh
ENTRYPOINT ["/bin/sh", "/entrypoint.sh"]
