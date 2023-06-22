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

from clip_benchmark.cli import TARGET_JSON


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
    datasets = args.dataset.split(",")
    split = args.split
    task = "zeroshot_classification"
    language = "en"
    output_dir = "output"
    debugstr = ""
    target_json_new = TARGET_JSON

    model_re = None
    if args.model_regex != "":
        model_re = re.compile(args.model_regex)

    # model_list = open_clip.pretrained.list_pretrained()
    model_list = [
        ("ViT-L-14", "openai"),  # 0.43G
        ("EVA02-L-14-336", "merged2b_s6b_b61k"),  # 0.43G
        ("ViT-bigG-14", "laion2b_s39b_b160k"),  # 2.54G
        ("EVA02-E-14-plus", "laion2b_s9b_b144k"),  # 5.04G
    ]
    # template_list = ["none", "imagenet1k", "caltech101"]
    template_list = ["imagenet1k"]

    transform_list = ["default", "resizeonly", "zeropad"]

    for model, pretrained in model_list:
        if model_re is not None:
            match = model_re.match(model)
            if (match and args.invert_regex) or (not match and not args.invert_regex):
                print(f"Regex IGNORE: {model} {pretrained}")
                continue
            else:
                print(f"Regex PASS:   {model} {pretrained}")
        pretrained_slug = os.path.basename(pretrained) if os.path.isfile(
            pretrained) else pretrained

        for template in template_list:
            batch_size = args.batch_size
            for dataset in datasets:
                dataset_slug = dataset.replace('/', '_')
                for transform in transform_list:
                    actual_json = target_json_new.format(
                        output_dir=output_dir,
                        debugstr=debugstr,
                        dataset=dataset_slug,
                        split=split,
                        pretrained=pretrained_slug,
                        model=model,
                        task=task,
                        language=language,
                        template=template,
                        transform=transform,
                    )

                    # print(f"Check for {actual_json}")
                    if os.path.exists(actual_json):
                        # print(f"Skipping {actual_json} since it already exists!")
                        continue
                    else:
                        pass
                    cmd = (f"python -m clip_benchmark.cli eval "
                           f"--dataset {dataset} --split {split} "
                           f"--model {model} --pretrained {pretrained} "
                           f"--task zeroshot_classification "
                           f"--template {template} --transform {transform} "
                           f"--batch_size {batch_size} --skip_existing")
                    print("#" * 80)
                    print(cmd)
                    print("#" * 80)
                    if not args.test:
                        os.system(cmd)


if __name__ == "__main__":
    main()
