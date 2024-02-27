import argparse

import cv2
import torch
from image_matcher import ImageMatcher
from loguru import logger


def get_args():
    parser = argparse.ArgumentParser("test matching two images")
    parser.add_argument("--query_path", type=str, default="./data/car1.jpg")
    parser.add_argument("--ref_path", type=str, default="./data/car2.jpg")
    parser.add_argument("--draw_num", type=int, default=20)

    return parser.parse_args()


def main():
    args = get_args()
    _image_matcher = ImageMatcher(
        device_str="cpu" if not torch.cuda.is_available() else "cuda",
    )
    image_paths = [
        args.query_path,
        args.ref_path,
    ]
    images = [cv2.imread(image_path, 0) for image_path in image_paths]
    for image, image_path in zip(images, image_paths):
        assert image is not None, f"invalid image path {image_path}"

    query_kpts, ref_kpts = _image_matcher.run(*images)

    logger.info(f"matching {args.query_path} to {args.ref_path}, number of matches {len(query_kpts)}")
    match_img = cv2.drawMatches(
        images[0],
        query_kpts[: args.draw_num],
        images[1],
        ref_kpts[: args.draw_num],
        [cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=0) for i in range(args.draw_num)],
        None,
        flags=2,
    )
    cv2.imwrite("match_img.jpg", match_img)


if __name__ == "__main__":
    main()
