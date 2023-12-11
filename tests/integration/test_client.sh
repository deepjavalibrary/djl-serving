#!/usr/bin/env bash

content_type=$1
content=$2
url=$3

if [[ -z "$url" ]]; then
  url="http://127.0.0.1:8080/predictions/test"
fi

echo "Testing $url with content type: $content_type ..."

if [[ "$content_type" == "tensor/ndlist" ]]; then
  djl-bench ndlist-gen -s $content -o test.ndlist
  curl -sf -m 10 -X POST $url -o out.ndlist -T test.ndlist -H "Content-type: $content_type"
  ret=$?
  rm -rf test.ndlist out.ndlist
elif [[ "$content_type" == "tensor/npz" ]]; then
  djl-bench ndlist-gen -s $content -z -o test.npz
  curl -sf -m 10 -X POST $url -o out.npz -T test.npz -H "Content-type: $content_type"
  ret=$?
  rm -rf test.npz out.npz
elif [[ "$content_type" == "text/plain" ]]; then
  curl -sf -m 10 -X POST $url -d "$content" -H "Content-type: $content_type"
  ret=$?
elif [[ "$content_type" == "image/jpg" ]]; then
  curl -sf -m 10 -X POST $url -T $content -H "Content-type: $content_type"
  ret=$?
else
  echo "Content type $content_type not supported!"
  exit 1
fi

if [[ -z "$EXPECT_TIMEOUT" ]]; then
  if [[ $ret -ne 0 ]]; then
    echo "Request failed: $ret"
    exit 1
  fi
else
  if [[ $ret -ne 28 ]]; then
    echo "Expecting time out, actual: $ret"
    exit 1
  fi
fi

