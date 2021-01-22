
docker run --volume $(pwd):/test --user $(id -u):$(id -g) gsplines bash test.sh
docker run --volume $(pwd):/test --user $(id -u):$(id -g) gsplines \
    pip3 install --user git+https://github.com/rafaelrojasmiliani/gsplines.git
