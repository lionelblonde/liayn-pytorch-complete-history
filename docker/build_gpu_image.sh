# Build docker image
docker build --no-cache -f Dockerfile.gpu -t $1 .
