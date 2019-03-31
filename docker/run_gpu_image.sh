cd ..

docker run -i -t --rm -v $(pwd)/data:/code/data \
                      -v $(pwd)/algorithms:/code/algorithms \
                      -v $(pwd)/launchers:/code/launchers ad2d-pytorch-gpu:latest bash
