import oyaml as yaml
import os, sys
from PIL import Image, ImageFont, ImageDraw
from loguru import logger
from tqdm.auto import tqdm
from argparse import ArgumentParser


Font = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
if os.name == 'nt':
    Font = "arial.ttf"


def concat_images(img_list):
    "img_list: 2 dim array, 0 dim - images in rows, 1 dim - images in columns"
    cols = max([len(i) for i in img_list])
    rows = len(img_list)

    composite = Image.new('RGB', (1030 * cols - 6, 1030 * rows - 6))
    for r in range(rows):
        for c in range(len(img_list[r])):
            composite.paste(Image.open(img_list[r][c]), (1030 * c, 1030 * r))
    return composite

def add_titles_concat_images(titles, concat_images):
    side = Image.new('RGB', (700 + concat_images.width, 1030 * len(titles) - 6))
    draw = ImageDraw.Draw(side)
    font = ImageFont.truetype(Font, 80)
    for i, title in enumerate(titles):
        draw.text((5, 1030 * i + 1030 / 2), title, (255,255,255), font=font)
    side.paste(concat_images, (700, 0))
    return side

def matrix_titles_images(row_titles, col_titles, concat_images, row_name="", col_name=""):
    side = Image.new('RGB', (500 + concat_images.width, 100 + concat_images.height))
    draw = ImageDraw.Draw(side)
    font = ImageFont.truetype(Font, 70, encoding="unic")
    draw.text((5, 200), str(row_name) + ":", (255,255,255), font=font)
    draw.text((300, 15), str(col_name) + ":", (255,255,255), font=font)
    for i, title in enumerate(row_titles):
        draw.text((10, 1030 * i + 1030 / 2 + 100), str(title), (255,255,255), font=font)
    for i, title in enumerate(col_titles):
        draw.text((1030 * i + 1030 / 2 + 500, 15), str(title), (255,255,255), font=font)
    side.paste(concat_images, (500, 100))
    return side



def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("training_config", help="path to training config to get parameters")

    return parser.parse_args()




if __name__ == "__main__":
    args = _parse_args()

    with open(args.training_config, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader).get('config', {})

    matrix_config = config['process'][0].get('matrix_params', {})
    if not matrix_config.get('generate_matrix', False):
        exit()


    from flux_pipeline import FluxPipeline

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    logger.remove(0)
    logger.add(sys.stderr, level="ERROR")


    # -------------- Matrix generation
    print('Loading model')
    pipe = FluxPipeline.load_pipeline_from_config_path("configs/config-dev-H100.json")
    # pipe = FluxPipeline.load_pipeline_from_config_path("configs/config-dev-offload-3-4090.json")
    # character Lora
    lora_path = os.path.join(config['process'][0]['training_folder'], config['name'], config['name'] + '.safetensors')
    trigger = config['process'][0]['trigger_word']
    pipe.model.load_lora(lora_path, 1.0, "User")

    prompts = matrix_config['prompts']
    prompts = [p.replace('[trigger]', trigger) for p in prompts]
    adapter_names = []
    gen_images = []

    prog = tqdm(total=len(prompts) * len(matrix_config['adapter_paths']))
    for adap in range(len(matrix_config['adapter_paths'])):
        adapter_images = []
        path = matrix_config['adapter_paths'][adap]
        name = os.path.basename(path).split(('.safet'))[0]
        adap_trigger_path = os.path.join(os.path.dirname(path), name + '.txt')
        adap_trigger = ""
        if os.path.exists(adap_trigger_path):
            with open(adap_trigger_path, 'r') as f:
                adap_trigger = f.read()

        pipe.model.load_lora(path, 1.0, name, silent=True)
        adapter_names.append(name)
        prog.set_description(name)

        for prompt in prompts:
            prompt = adap_trigger + ' ' + prompt
            output_jpeg_bytes = pipe.generate(prompt=prompt, width=matrix_config['width'], height=matrix_config['height'],
                                              num_steps=matrix_config['sample_steps'], guidance=4, seed=None, silent=True)
            adapter_images.append(output_jpeg_bytes)
            prog.update()
        pipe.unload_lora(name, silent=True)
        gen_images.append(adapter_images)

    ci = concat_images(gen_images)
    tci = add_titles_concat_images(adapter_names, ci)

    tci.save(os.path.join(config['process'][0]['training_folder'], config['name'], "matrix_adapters.jpg"), "JPEG", quality=95)



