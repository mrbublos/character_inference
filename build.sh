export VERSION=1.32
docker buildx build --platform linux/amd64 -t skrendelauth/inference:$VERSION -f docker/Dockerfile .
docker push skrendelauth/inference:$VERSION