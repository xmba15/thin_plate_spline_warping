import argparse
from typing import List

import cv2
import numpy as np
import torch
from image_matcher import ImageMatcher
from loguru import logger
from thin_plate_spline import ThinPlateSpline


def get_args():
    parser = argparse.ArgumentParser("test matching two images")
    parser.add_argument("--query_path", type=str, default="./data/car1.jpg")
    parser.add_argument("--ref_path", type=str, default="./data/car2.jpg")
    parser.add_argument("--grid_size", type=int, default=50)

    return parser.parse_args()


def _to_kpts_arr(cv2_kpts: List[cv2.KeyPoint]):
    return np.array([kpt.pt for kpt in cv2_kpts])


def main():
    args = get_args()
    image_paths = [
        args.query_path,
        args.ref_path,
    ]
    images = [cv2.imread(image_path) for image_path in image_paths]
    for image, image_path in zip(images, image_paths):
        assert image is not None, f"invalid image path {image_path}"

    query_kpts, ref_kpts = ImageMatcher(
        config={"grid_size": args.grid_size},
        device_str="cpu" if not torch.cuda.is_available() else "cuda",
    ).run(*[cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images])
    logger.info(f"matching {args.query_path} to {args.ref_path}, number of matches {len(query_kpts)}")

    _tps_handler = ThinPlateSpline()

    # as cv2.remap is backward mapping so need to fit a tps that maps ref kpts to query kpts
    _tps_handler.fit(
        _to_kpts_arr(ref_kpts),
        _to_kpts_arr(query_kpts),
    )
    height, width = images[1].shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    _grid = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    warped_grid = _tps_handler.predict(_grid).reshape(height, width, 2).astype(np.float32)
    warped_image = cv2.remap(
        images[0],
        warped_grid[:, :, 0],
        warped_grid[:, :, 1],
        interpolation=cv2.INTER_CUBIC,
    )
    cv2.imwrite("warped_image.jpg", warped_image)
    cv2.imwrite("merged_image.jpg", cv2.addWeighted(images[1], 0.5, warped_image, 0.5, 0.0))


if __name__ == "__main__":
    main()
