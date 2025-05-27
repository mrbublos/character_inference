export VERSION=1.17
docker buildx build --platform linux/amd64 -t skrendelauth/inference:$VERSION -f Dockerfile .
docker push skrendelauth/inference:$VERSION

#bazel build //:inference_embedded
