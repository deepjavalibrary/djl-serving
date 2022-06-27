#!/usr/bin/env bash

set -e

url=$1
content_type=$2
content=$3

echo "Testing $url with content type: $content_type ..."

if [[ "$content_type" == "tensor/ndlist" ]]; then
  djl-bench ndlist-gen -s $content -o test.ndlist
  curl -f -X POST $url \
    -T test.ndlist \
    -H "Content-type: $content_type"
  rm -rf test.ndlist
elif [[ "$content_type" == "tensor/npz" ]]; then
  djl-bench ndlist-gen -s $content -z -o test.npz
  curl -f -X POST $url \
    -T test.npz \
    -H "Content-type: $content_type"
  rm -rf test.npz
elif [[ "$content_type" == "text/plain" ]]; then
  curl -f -X POST $url \
    -d "$content" \
    -H "Content-type: $content_type"
elif [[ "$content_type" == "image/jpg" ]]; then
  curl $content -o test.jpg
  curl -f -X POST $url \
    -T test.jpg \
    -H "Content-type: $content_type"
  rm -rf test.jpg
else
  echo "Content type $content_type not supported!"
  exit 1
fi
