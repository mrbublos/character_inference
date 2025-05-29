apt update && apt install -y sudo
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh

wget https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
sudo cp ./bazelisk-linux-amd64 /usr/local/bin/bazel

git clone https://github.com/mrbublos/character_inference.git

cd character_inference

apt-get install rsync -y

rsync -ah --progress /workspace/ht/models--black-forest-labs--flux.1-dev hf/model/
rsync -ah --progress /workspace/lora_styles hf/styles

docker login -u skrendelauth

bazel build //:push_inference_embedded
