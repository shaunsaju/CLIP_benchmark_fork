"""Console script for clip_benchmark."""
import argparse
import csv
import glob
import json
import os
import sys
from copy import copy, deepcopy
from pathlib import Path
from pprint import pprint

import joblib
import torch
import torchvision
from torchvision.transforms import InterpolationMode

from clip_benchmark.datasets.builder import build_dataset, get_dataset_collate_fn,\
    get_dataset_default_task, dataset_collection, get_dataset_collection_from_file
from clip_benchmark.metrics import zeroshot_classification, zeroshot_retrieval, linear_probe,\
    mscoco_generative
from clip_benchmark.model_collection import get_model_collection_from_file, model_collection
from clip_benchmark.models import load_clip, MODEL_TYPES


TARGET_JSON="{output_dir}/{debugstr}{dataset}~{split}~{pretrained}~{model}~{language}~{task}~{template}~{transform}/result.json"

def get_parser_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_eval = subparsers.add_parser('eval', help='Evaluate')
    parser_eval.add_argument('--dataset', type=str, default="cifar10", nargs="+",
                             help="Dataset(s) to use for the benchmark. Can be the name of a dataset, or a collection name ('vtab', 'vtab+', 'imagenet_robustness', 'retrieval') or path of a text file where each line is a dataset name")
    parser_eval.add_argument('--dataset_root', default=None, type=str,
                             help="dataset root folder where the datasets are downloaded. Can be in the form of a template depending on dataset name, e.g., --dataset_root='datasets/{dataset}'. This is useful if you evaluate on multiple datasets.")
    parser_eval.add_argument('--split', type=str, default="test", help="Dataset split to use")
    parser_eval.add_argument('--model', type=str, default="ViT-B-32-quickgelu",
                             help="Model architecture to use from OpenCLIP")
    parser_eval.add_argument('--pretrained', type=str, default="laion400m_e32",
                             help="Model checkpoint name to use from OpenCLIP")
    parser_eval.add_argument('--pretrained_model', type=str, default="", nargs="+",
                             help="Pre-trained model(s) to use. Can be the full model name where `model` and `pretrained` are comma separated (e.g., --pretrained_model='ViT-B-32-quickgelu,laion400m_e32'), a model collection name ('openai' or 'openclip_base' or 'openclip_multilingual' or 'openclip_all'), or path of a text file where each line is a model fullname where model and pretrained are comma separated (e.g., ViT-B-32-quickgelu,laion400m_e32). --model and --pretrained are ignored if --pretrained_model is used.")
    parser_eval.add_argument('--task', type=str, default="auto",
                             choices=["zeroshot_classification", "zeroshot_retrieval",
                                      "linear_probe", "mscoco_generative", "auto"],
                             help="Task to evaluate on. With --task=auto, the task is automatically inferred from the dataset.")
    parser_eval.add_argument('--no_amp', action="store_false", dest="amp", default=True,
                             help="whether to use mixed precision")
    parser_eval.add_argument('--num_workers', default=4, type=int)
    parser_eval.add_argument('--recall_k', default=[5], type=int,
                             help="for retrieval, select the k for Recall@K metric. ", nargs="+", )
    parser_eval.add_argument('--fewshot_k', default=-1, type=int,
                             help="for linear probe, how many shots. -1 = whole dataset.")
    parser_eval.add_argument('--fewshot_epochs', default=10, type=int,
                             help="for linear probe, how many epochs.")
    parser_eval.add_argument('--fewshot_lr', default=0.1, type=float,
                             help="for linear probe, what is the learning rate.")
    parser_eval.add_argument("--skip_load", action="store_true",
                             help="for linear probes, when everything is cached, no need to load model.")
    parser_eval.add_argument('--seed', default=0, type=int, help="random seed.")
    parser_eval.add_argument('--batch_size', default=64, type=int)
    parser_eval.add_argument('--model_cache_dir', default=None, type=str,
                             help="directory to where downloaded models are cached")
    parser_eval.add_argument('--feature_root', default="features", type=str,
                             help="feature root folder where the features are stored.")
    parser_eval.add_argument('--annotation_file', default="", type=str,
                             help="text annotation file for retrieval datasets. Only needed  for when `--task` is `zeroshot_retrieval`.")
    parser_eval.add_argument('--language', default="en", type=str, nargs="+",
                             help="language(s) of classname and prompts to use for zeroshot classification.")
    parser_eval.add_argument('--output_dir', default="output")
    parser_eval.add_argument('--output', default=TARGET_JSON, type=str,
                             help="output file where to dump the metrics. Can be in form of a template, e.g., --output='{dataset}_.....json'")  # {pretrained_full_path}_ not necessary
    parser_eval.add_argument('--quiet', dest='verbose', action="store_false",
                             help="suppress verbose messages")
    parser_eval.add_argument('--cupl', default=False, action="store_true",
                             help="Use natural language prompt from CuPL paper")
    parser_eval.add_argument('--save_clf', default=None, type=str,
                             help="optionally save the classification layer output by the text tower")
    parser_eval.add_argument('--load_clfs', nargs='+', default=[], type=str,
                             help="optionally load and average mutliple layers output by text towers.")
    parser_eval.add_argument('--skip_existing', default=False, action="store_true",
                             help="whether to skip an evaluation if the output file exists.")
    parser_eval.add_argument('--model_type', default="open_clip", type=str, choices=MODEL_TYPES,
                             help="clip model type")
    parser_eval.add_argument('--wds_cache_dir', default=None, type=str,
                             help="optional cache directory for webdataset only")
    parser_eval.add_argument('--small', type=int, default=0,
                             help="Creater smaller version of dataset for testing")
    parser_eval.add_argument('--template', type=str, default="auto",
                             help="Fix used template instead of using dataset name")
    parser_eval.add_argument('--transform', type=str, default="default",
                             help="use another transform")
    parser_eval.set_defaults(which='eval')

    parser_build = subparsers.add_parser('build', help='Build CSV from evaluations')
    parser_build.add_argument('files', type=str, nargs="+", help="path(s) of JSON result files")
    parser_build.add_argument('--output', type=str, default="benchmark.csv", help="CSV output file")
    parser_build.add_argument("-t", "--test", action="store_true", help="Test mode (no write)")
    parser_build.set_defaults(which='build')

    parser_build = subparsers.add_parser('globbuild', help='Search and build CSV from evaluations')
    parser_build.add_argument('glob', type=str, help="Glob path(s) of JSON result files")
    parser_build.add_argument('--output', type=str, default="benchmark.csv", help="CSV output file")
    parser_build.add_argument("-t", "--test", action="store_true", help="Test mode (no write)")
    parser_build.set_defaults(which='globbuild')

    args = parser.parse_args()
    return args, parser


def main():
    base, parser = get_parser_args()
    if "which" not in base:
        parser.print_help()
        sys.exit(1)
    if base.which == "eval":
        main_eval(base)
    elif base.which == "build":
        main_build(base)
    elif base.which == "globbuild":
        base.files = sorted(list(glob.glob(base.glob, recursive=True)))
        if len(base.files) == 0:
            print(f"No files found for {base.glob}")
            sys.exit(1)
        pprint(base.files)
        main_build(base)


def main_build(base):
    # Build a benchmark single CSV file from a set of evaluations (JSON files)
    rows = []
    fieldnames = set()
    for path in base.files:
        data = json.load(open(path, "r", encoding="utf-8"))
        row = {}
        row.update(data["metrics"])
        row.update(data)
        del row["metrics"]
        # row['model_fullname'] = row['model'] + ' ' + row['pretrained']  # not necessary

        # missing information: split, prompt
        # vic_caltech101~val~yfcc15m~RN50-quickgelu~en~zeroshot_classification~none
        dataset, split, pretrained, model, language, task, template, transform = Path(
            path).parent.as_posix().split("~")
        row["split"] = split
        row["template"] = template
        row["transform"] = transform

        # load model size from file
        size_file = Path(f"model_sizes/{model}~{pretrained}.json")
        params_g = json.load(size_file.open("r", encoding="utf-8"))["params_g"]
        row["gigaparams"] = params_g

        # dict_keys(['acc1', 'acc5', 'mean_per_class_recall', 'dataset', 'model', 'pretrained', 'task', 'language'])
        for field in row.keys():
            fieldnames.add(field)
        rows.append(row)
    if base.test:
        pprint(rows)
        print(f"Not writing to {base.output} in test mode")
        return
    print(f"Writing to {base.output}")

    # properly sort field names
    sorted_fieldnames = []
    sort_order = ["language", "task", "dataset", "split", "pretrained", "model", "gigaparams",
                  "template", "transform"]
    for field in sort_order:
        sorted_fieldnames.append(field)
        fieldnames.remove(field)
    for remaining_field in sorted(fieldnames):
        sorted_fieldnames.append(remaining_field)

    with open(base.output, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main_eval(base):
    # Get list of pre-trained models to evaluate
    pretrained_model = _as_list(base.pretrained_model)
    if pretrained_model:
        models = []
        for name in pretrained_model:
            if os.path.isfile(name):
                # if path, read file, each line is a pre-trained model
                models.extend(get_model_collection_from_file(name))
            elif name in model_collection:
                # if part of `model_collection`, retrieve from it
                models.extend(model_collection[name])
            else:
                # if not, assume it is in the form of `model,pretrained`
                model, pretrained = name.split(',')
                models.append((model, pretrained))
    else:
        models = [(base.model, base.pretrained)]

    # Ge list of datasets to evaluate on
    datasets = []
    for name in _as_list(base.dataset):
        if os.path.isfile(name):
            # If path, read file, each line is a dataset name
            datasets.extend(get_dataset_collection_from_file(name))
        elif name in dataset_collection:
            # if part of `dataset_collection`, retrieve from it
            datasets.extend(dataset_collection[name])
        else:
            # if not, assume it is simply the name of the dataset
            datasets.append(name)

    # Get list of languages to evaluate on
    languages = _as_list(base.language)

    if base.verbose:
        print(f"Models: {models}")
        print(f"Datasets: {datasets}")
        print(f"Languages: {languages}")

    for model, pretrained in models:
        for dataset in datasets:
            for language in languages:
                # We iterative over all possible model/dataset/languages
                # TODO: possibility to parallelize evaluation here
                args = copy(base)
                args.model = model
                args.pretrained = pretrained
                args.dataset = dataset
                args.language = language
                run(args)


def _as_list(l):
    if not l:
        return []
    return [l] if type(l) != list else l


def run(args):
    """Console script for clip_benchmark."""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # set seed.
    torch.manual_seed(args.seed)
    task = args.task
    if args.dataset.startswith("wds/"):
        dataset_name = args.dataset.replace("wds/", "", 1)
    else:
        dataset_name = args.dataset
    if task == "auto":
        task = get_dataset_default_task(dataset_name)
    pretrained_slug = os.path.basename(args.pretrained) if os.path.isfile(
        args.pretrained) else args.pretrained
    pretrained_slug_full_path = args.pretrained.replace('/', '_') if os.path.isfile(
        args.pretrained) else args.pretrained
    dataset_slug = dataset_name.replace('/', '_')
    output = args.output.format(
        output_dir=args.output_dir,
        debugstr=f"DEBUGmax{args.small}-" if args.small > 0 else "",
        dataset=dataset_slug,
        split=args.split,
        pretrained=pretrained_slug,
        pretrained_full_path=pretrained_slug_full_path,
        model=args.model,
        task=task,
        language=args.language,
        template=args.template,
        transform=args.transform,
    )
    if os.path.exists(output) and args.skip_existing:
        if args.verbose:
            print(f"Skip {output}, exists already.")
        return
    if args.verbose:
        print(
            f"Running '{task}' on '{dataset_name}' split {args.split} with the model '{args.pretrained}' on language '{args.language}'")

    dataset_root = args.dataset_root
    if dataset_root is None:
        if "CV_DATA_DIR" not in os.environ:
            raise ValueError("Either set envvar CV_DATA_DIR to the root of the dataset folder "
                             "or pass --dataset_root")
        dataset_root = (
                Path(os.environ["CV_DATA_DIR"]) / "clip_benchmark/{dataset_cleaned}").as_posix()
    dataset_root = dataset_root.format(dataset=dataset_name,
                                       dataset_cleaned=dataset_name.replace("/", "-"))
    print(f"Dataset root: {dataset_root}")
    if args.skip_load:
        model, transform, collate_fn, dataloader = None, None, None, None
    else:
        model, transform, tokenizer = load_clip(
            model_type=args.model_type,
            model_name=args.model,
            pretrained=args.pretrained,
            cache_dir=args.model_cache_dir,
            device=args.device
        )
        model.eval()

        if args.transform != "default":
            # assuming its a compose of 5 transforms

            # default: ratio-preserving resize shorter side to 224, then centercrop
            assert isinstance(transform, torchvision.transforms.Compose)
            old_transforms = transform.transforms
            assert len(old_transforms) == 5
            assert isinstance(old_transforms[0], torchvision.transforms.Resize)
            size = old_transforms[0].size
            assert isinstance(size, int)
            assert isinstance(old_transforms[1], torchvision.transforms.CenterCrop)

            if args.transform == "resizeonly":
                # ignore ratio-preserving and just resize to 224x224
                new_transforms = [
                    torchvision.transforms.Resize(
                        (size, size), interpolation=InterpolationMode.BICUBIC),
                ]
            elif args.transform == "zeropad":
                # ratio-preserving size longer side to 224, then zeropad by using centercrop
                new_transforms = [
                    torchvision.transforms.Resize(
                        size - 1, max_size=size, interpolation=InterpolationMode.BICUBIC),
                    torchvision.transforms.CenterCrop(
                        (size, size)),
                ]
            else:
                raise ValueError(f"Unknown custom transform {args.transform}")
            new_transform = torchvision.transforms.Compose([
                *new_transforms,
                *[deepcopy(x) for x in old_transforms[2:]],
            ])
            print(f"Replacing transform {transform} with {new_transform}")
            transform = new_transform

        dataset = build_dataset(
            dataset_name=args.dataset,
            root=dataset_root,
            transform=transform,
            split=args.split,
            annotation_file=args.annotation_file,
            download=True,
            language=args.language,
            task=task,
            cupl=args.cupl,
            wds_cache_dir=args.wds_cache_dir,
            small=args.small,
            template=args.template,
        )
        collate_fn = get_dataset_collate_fn(args.dataset)
        if args.verbose:
            try:
                print(f"Dataset size: {len(dataset)}")
            except TypeError:
                print("IterableDataset has no len()")
            print(f"Dataset split: {args.split}")
            if dataset.classes:
                try:
                    print(f"Dataset classes: {dataset.classes}")
                    print(f"Dataset number of classes: {len(dataset.classes)}")
                except AttributeError:
                    print("Dataset has no classes.")

        if args.dataset.startswith("wds/"):
            dataloader = torch.utils.data.DataLoader(
                dataset.batched(args.batch_size), batch_size=None,
                shuffle=False, num_workers=args.num_workers,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                collate_fn=collate_fn
            )

    # create unique hash for this experiment
    hash_list = []
    for args_field in ["dataset", "language", "model", "model_type", "pretrained",
                       "pretrained_model", "task", "which", "template"]:
        hash_list.append(getattr(args, args_field))
    args_hash = joblib.hash(hash_list)

    logits, target = None, None
    if task == "zeroshot_classification":
        zeroshot_templates = dataset.templates if hasattr(dataset, "templates") else None
        if args.cupl:
            assert (zeroshot_templates is not None), "Dataset does not support CuPL prompts"
        if args.verbose:
            print(f"Zero-shot templates: {zeroshot_templates}")
        classnames = dataset.classes if hasattr(dataset, "classes") else None
        assert (
                zeroshot_templates is not None and classnames is not None), "Dataset does not support classification"
        metrics, logits, target = zeroshot_classification.evaluate(
            model,
            dataloader,
            tokenizer,
            classnames, zeroshot_templates,
            device=args.device,
            amp=args.amp,
            verbose=args.verbose,
            cupl=args.cupl,
            save_clf=args.save_clf,
            load_clfs=args.load_clfs,
            args_hash=args_hash,
        )
    elif task == "zeroshot_retrieval":
        metrics = zeroshot_retrieval.evaluate(
            model,
            dataloader,
            tokenizer,
            recall_k_list=args.recall_k,
            device=args.device,
            amp=args.amp
        )
    elif task == "linear_probe":
        # we also need the train split for linear probing.
        train_dataset = build_dataset(
            dataset_name=args.dataset,
            root=dataset_root,
            transform=transform,
            split='train',
            annotation_file=args.annotation_file,
            download=True,
            small=args.small,
            template=args.template,
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            collate_fn=collate_fn, pin_memory=True,
        )
        metrics = linear_probe.evaluate(
            model,
            train_dataloader,
            dataloader,
            args.fewshot_k,
            args.batch_size,
            args.num_workers,
            args.fewshot_lr,
            args.fewshot_epochs,
            (args.model + '-' + args.pretrained + '-' + args.dataset).replace('/', '_'),
            args.seed,
            args.feature_root,
            device=args.device,
            amp=args.amp,
            verbose=args.verbose,
        )
    elif task == "mscoco_generative":
        metrics = mscoco_generative.evaluate(
            model=model,
            dataloader=dataloader,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            amp=args.amp,
            verbose=args.verbose,
            transform=transform
        )
    else:
        raise ValueError(
            "Unsupported task: {}. task should `zeroshot_classification` or `zeroshot_retrieval`".format(
                task))
    dump = {
        "dataset": args.dataset,
        "model": args.model,
        "pretrained": args.pretrained,
        "task": task,
        "metrics": metrics,
        "language": args.language,
    }
    if args.verbose:
        print(f"Dump results to: {output}")
    output_json = Path(output)
    os.makedirs(output_json.parent, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dump, f)
    if logits is not None:
        output_logits = output_json.parent / (output_json.stem + "_logits.pt")
        torch.save(logits.cpu(), output_logits)
        # possible improvements:
        #   saving as half or 8bit could save space.
        #   saving in e.g. h5 could be faster to access single datapoints.
        #   save only top50 instead of all 1000 logits.
    if target is not None:
        output_target = output_json.parent / (output_json.stem + "_target.pt")
        torch.save(target.cpu(), output_target)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
