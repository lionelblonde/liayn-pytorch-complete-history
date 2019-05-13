cd ..

docker run -i -t --rm -v $(pwd)/data:/code/data \
                      -v $(pwd)/algorithms:/code/algorithms \
                      -v $(pwd)/launchers:/code/launchers $1:latest bash
