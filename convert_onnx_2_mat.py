#!/usr/bin/env python3
# example: python3 convert_onnx_2_mat.py -i simple_rnn.onnx -o repaired_model1.mat
from typing import AnyStr

import click
import onnx
from onnx import numpy_helper
from scipy.io import savemat


def convert_onnx_to_mat(input_path: AnyStr, output_path: AnyStr) -> None:
    model = onnx.load(input_path)
    params = {
        t.name: numpy_helper.to_array(t) for t in model.graph.initializer
    }
    savemat(output_path, params)


@click.command(
    help="Converts a .onnx file to a .mat file compatible with MIPVerify.jl"
)
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(dir_okay=False, writable=True),
    required=True,
)
def main(input_path, output_path):
    convert_onnx_to_mat(input_path, output_path)


if __name__ == "__main__":
    main()
