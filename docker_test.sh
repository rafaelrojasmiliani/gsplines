
docker run --volume $(pwd):/test --user $(id -u):$(id -g) gspline:tester bash test.sh
