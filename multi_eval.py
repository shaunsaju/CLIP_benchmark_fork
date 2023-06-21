"""
# to discard >24G requirement models
python multi_eval.py -m "EVA02-E.*" -i

"""
import os
from pathlib import Path
from pprint import pprint

import open_clip

from attr import define
from typedparser import VerboseQuietArgs, add_argument, TypedParser
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from loguru import logger
import re


@define
class Args(VerboseQuietArgs):
    test: bool = add_argument(shortcut="-t", action="store_true", help="Test only.")
    dataset: str = add_argument(
        shortcut="-d", type=str, help="Dataset name", default="vic/caltech101")
    split: str = add_argument(
        shortcut="-s", type=str, help="Dataset split", default="train")
    model_regex: str = add_argument(
        shortcut="-m", type=str, help="Model regex", default="")

    invert_regex: bool = add_argument(
        shortcut="-i", action="store_true", help="Invert regex.")
    batch_size: int = add_argument(
        shortcut="-b", type=int, help="Batch size", default=64)


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.debug(open_clip.pretrained.list_pretrained())
    logger.info(args)

    root = os.environ["CV_DATA_DIR"]
    os.makedirs(root, exist_ok=True)
    dataset = args.dataset
    split = args.split
    task = "zeroshot_classification"
    language = "en"
    output_dir = "output"
    debugstr = ""
    dataset_slug = dataset.replace('/', '_')
    target_json_new = "{output_dir}/{debugstr}{dataset}~{split}~{pretrained}~{model}~{language}~{task}{templatestr}/result.json"

    model_re = None
    if args.model_regex != "":
        model_re = re.compile(args.model_regex)

    model_list = open_clip.pretrained.list_pretrained()
    model_list = [
        ("ViT-L-14", "openai"),
        ("EVA02-L-14-336", "merged2b_s6b_b61k"),
        ("EVA02-L-14-336", "merged2b_s6b_b61k"),
        ("EVA02-E-14-plus", "laion2b_s9b_b144k"),
    ]

    for model, pretrained in model_list:
        # for model, pretrained in [
        #     ("ViT-bigG-14", "laion2b_s39b_b160k"),
        #     ("ViT-L-14", "openai"),
        #     ("ViT-L-14", "laion2b_s32b_b82k"),
        #     ("ViT-L-14", "datacomp_xl_s13b_b90k"),
        # ]:
        if model_re is not None:
            match = model_re.match(model)
            if (match and args.invert_regex) or (not match and not args.invert_regex):
                print(f"Regex IGNORE: {model} {pretrained}")
                continue
            else:
                print(f"Regex PASS:   {model} {pretrained}")

        for template_override in ["none", "imagenet1k", "caltech101"]:
            batch_size = args.batch_size

            templatestrnew = f"~{template_override}" if template_override is not None else ""
            actual_json = target_json_new.format(
                output_dir=output_dir,
                debugstr=debugstr,
                dataset=dataset_slug,
                split=split,
                pretrained=pretrained,
                model=model,
                language=language,
                task=task,
                templatestr=templatestrnew,
            )

            # print(f"Check for {actual_json}")
            if os.path.exists(actual_json):
                # print(f"Skipping {actual_json} since it already exists!")
                continue
            else:
                pass
                # print(f"TODO {pretrained} {model}")
                # continue
            cmd = (f"python -m clip_benchmark.cli eval --model {model} --pretrained {pretrained} "
                   f"--task zeroshot_classification --dataset {dataset} --split {split} "
                   f"--template_override {template_override} --skip_existing "
                   f"--dataset_root {root}/clip_benchmark/{dataset} --batch_size {batch_size}")
            print("#" * 80)
            print(cmd)
            print("#" * 80)
            if not args.test:
                os.system(cmd)


if __name__ == "__main__":
    main()
