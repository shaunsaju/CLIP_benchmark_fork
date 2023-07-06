import json
import os
from pathlib import Path
from typing import Dict

import open_clip
from attr import define
from loguru import logger
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import VerboseQuietArgs, TypedParser


@define
class Args(VerboseQuietArgs):
    pass


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args))

    output_dir = Path("model_sizes")
    os.makedirs(output_dir, exist_ok=True)
    for model_name, pretrained in open_clip.pretrained.list_pretrained():
        output_file = output_dir / f"{model_name}~{pretrained}.json"
        if output_file.is_file():
            continue

        model = open_clip.create_model(model_name, pretrained=pretrained, device="cpu")
        params_dict = count_params(model)
        params_sum = sum(params_dict.values())
        params_g = params_sum / 1e9
        logger.info(f"Model: {model_name} | Pretrained: {pretrained} | Params: {params_g:.3f}G")
        with output_file.open("w", encoding="utf-8") as f:
            json.dump({"params_g": params_g}, f, indent=2)
        del model


def count_params(parameters) -> Dict[str, int]:
    # support inputs: model, model.parameters(), model.named_parameters()
    if hasattr(parameters, "values"):
        parameters = parameters.values()
    if hasattr(parameters, "parameters"):
        parameters = parameters.parameters()

    groups = {"grad": 0, "no_grad": 0}
    for v in parameters:
        if v.requires_grad:
            groups["grad"] += v.numel()
        else:
            groups["no_grad"] += v.numel()
    return groups


if __name__ == "__main__":
    main()
