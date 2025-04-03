import argparse
import uvicorn
from api_one_gpu import app, STYLES_FOLDER

import os


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Flux API server")
    parser.add_argument("-c", "--config-path", type=str, help="Path to the configuration file, if not provided, the model will be loaded from the command line arguments")
    parser.add_argument("-p", "--port", type=int, default=8088, help="Port to run the server on")
    parser.add_argument("-H", "--host", type=str, default="0.0.0.0", help="Host to run the server on")

    return parser.parse_args()


def main():
    args = parse_args()

    # lazy loading so cli returns fast instead of waiting for torch to load modules
    from flux_pipeline import FluxPipeline

    app.state.model = FluxPipeline.load_pipeline_from_config_path(args.config_path)
    print('warmup')
    warmup_dict = dict(
        prompt="A beautiful test image used to solidify the fp8 nn.Linear input scales prior to compilation 😉",
        height=768, width=768, num_steps=2, guidance=3.5, seed=10, )
    app.state.model.generate(**warmup_dict)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    os.makedirs(STYLES_FOLDER, exist_ok=True)
    main()
