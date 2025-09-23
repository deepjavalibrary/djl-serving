docker image rmi --force $(docker image ls -q)
rm -rf serving/docker/distributions
./gradlew clean && ./gradlew --refresh-dependencies :serving:dockerDeb -Psnapshot
cd serving/docker
export DJL_VERSION=$(awk -F '=' '/djl / {gsub(/ ?"/, "", $2); print $2}' ../../gradle/libs.versions.toml)
export SERVING_VERSION=$(awk -F '=' '/serving / {gsub(/ ?"/, "", $2); print $2}' ../../gradle/libs.versions.toml)
docker compose build --no-cache --build-arg djl_version=${DJL_VERSION} --build-arg djl_serving_version=${SERVING_VERSION} lmi
