#!/usr/bin/env vash

myuid=$(id -u $USER)
mygid=$(id -g $USER)
mygroup=$(id -g -n $USER)

remote=$(git config --get remote.origin.url)
scriptdir="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
docker build -t "gsplines:latest" \
    --build-arg myuser="$USER" \
    --build-arg myuid="$myuid" \
    --build-arg mygroup="$mygroup" \
    --build-arg mygid="$mygid" \
    --build-arg scriptdir="$scriptdir" \
    --build-arg remote="$remote" \
      -f ./testdocker.dockerfile .

exit 0

