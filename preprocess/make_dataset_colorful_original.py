from pathlib import Path

from gated_tree_clip.utils.clogging import getColoredLogger
from PIL import Image
from tqdm.auto import tqdm

logger = getColoredLogger(__name__)
logger.setLevel("INFO")
datasets_dir = Path.home() / "datasets"
target_dir = datasets_dir / "ColorfulTwoObject"
target_dir.mkdir(exist_ok=True, parents=True)

logger.info(f"{target_dir=}")

image_pathes = sorted(
    [p for d in (datasets_dir / "ColorfulOriginal").iterdir() for p in d.iterdir() if p.suffix in [".jpg", ".png"]]
)

for image_path_1 in tqdm(image_pathes.copy()):
    for image_path_2 in tqdm(image_pathes.copy(), leave=False):
        if image_path_1.parent != image_path_2.parent:
            canvas = Image.new("RGB", (224 * 2, 224))
            image_1 = Image.open(image_path_1).resize((224, 224))
            image_2 = Image.open(image_path_2).resize((224, 224))
            canvas.paste(image_1, (0, 0))
            canvas.paste(image_2, (224, 0))
            path = target_dir / f"{image_path_1.stem.lower()}_and_{image_path_2.stem.lower()}.jpg"
            canvas.save(path)


logger.info("Done!")
