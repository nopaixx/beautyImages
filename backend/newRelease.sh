#!/bin/bash
# docker login --username=
# $1 release id

docker build -t nopaixx/apiflask:$1 .
docker push nopaixx/apiflask:$1
