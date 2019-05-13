# Build docker image
docker build --no-cache -f Dockerfile.cpu -t $1 .
