.PHONY: build docker-test test

build:
\tdocker build -f docker/Dockerfile.inference -t base-nvidia-business:latest .

docker-up:
\tdocker-compose up -d

test:
\tpytest -q
