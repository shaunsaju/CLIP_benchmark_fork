import os
from pprint import pprint

import open_clip

pprint(open_clip.pretrained.list_pretrained())


def main():
    root = os.environ["CV_DATA_DIR"]
    assert os.path.exists(root), f"root={root} does not exist!"
    dataset = "vic/caltech101"
    split = "train"
    task = "zeroshot_classification"
    language = "en"
    output_dir = "output"
    debugstr = ""
    dataset_slug = dataset.replace('/', '_')
    target_json = "{output_dir}/{debugstr}{dataset}_{split}_{pretrained}_{model}_{language}_{task}{templatestr}/result.json"

    for model, pretrained in open_clip.pretrained.list_pretrained():
        # for model, pretrained in [
        #     ("ViT-bigG-14", "laion2b_s39b_b160k"),
        #     ("ViT-L-14", "openai"),
        #     ("ViT-L-14", "laion2b_s32b_b82k"),
        #     ("ViT-L-14", "datacomp_xl_s13b_b90k"),
        # ]:
        for template_override in ["none", "imagenet1k", "caltech101"]:
            templatestr = f"-{template_override}" if template_override is not None else ""
            actual_json = target_json.format(
                output_dir=output_dir,
                debugstr=debugstr,
                dataset=dataset_slug,
                split=split,
                pretrained=pretrained,
                model=model,
                language=language,
                task=task,
                templatestr=templatestr,
            )
            print(f"Check for {actual_json}")
            if os.path.exists(actual_json):
                print(f"Skipping {actual_json} since it already exists!")
                continue

            cmd = (f"python -m clip_benchmark.cli eval --model {model} --pretrained {pretrained} "
                   f"--task zeroshot_classification --dataset {dataset} --split {split} "
                   f"--template_override {template_override} --skip_existing "
                   f"--dataset_root {root}/clip_benchmark/{dataset} ")
            print("#" * 80)
            print(cmd)
            print("#" * 80)
            os.system(cmd)


if __name__ == "__main__":
    main()
