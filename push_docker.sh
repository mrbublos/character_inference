apt update && apt install -y sudo
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh

wget https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
sudo cp ./bazelisk-linux-amd64 /usr/local/bin/bazel

apt-get install rsync -y

cd /root && git clone https://github.com/mrbublos/character_inference.git
cd /root/character_inference && git checkout embeddedmodel

mkdir -p /root/character_inference/hf/model
mkdir -p /root/character_inference/hf/styles
rsync -ah --progress /workspace/hf/models--black-forest-labs--flux.1-dev /root/character_inference/hf/model/
rsync -ah --progress /workspace/lora_styles /root/character_inference/hf/styles/

docker login -u skrendelauth

cd /root/character_inference && bazel build //:push_inference_embedded
cd /root/character_inference && bazel run //:push_inference_embedded
