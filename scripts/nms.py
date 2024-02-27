"""
Efficient adaptive non-maximal suppression algorithms for homogeneous spatial keypoint distribution
Ref: https://github.com/BAILOOL/ANMS-Codes/blob/master/Python/ssc.py
"""

import math
from typing import List

import cv2


def ssc(
    keypoints: List[cv2.KeyPoint],
    num_ret_points: int,
    cols: int,
    rows: int,
    tolerance: float = 0.1,
) -> List[int]:
    exp1 = rows + cols + 2 * num_ret_points
    exp2 = (
        4 * cols
        + 4 * num_ret_points
        + 4 * rows * num_ret_points
        + rows * rows
        + cols * cols
        - 2 * rows * cols
        + 4 * rows * cols * num_ret_points
    )
    exp3 = math.sqrt(exp2)
    exp4 = num_ret_points - 1

    sol1 = -round(float(exp1 + exp3) / exp4)
    sol2 = -round(float(exp1 - exp3) / exp4)

    high: float = sol1 if (sol1 > sol2) else sol2
    low: float = math.floor(math.sqrt(len(keypoints) / num_ret_points))

    prev_width = -1.0
    indices_list = []
    result: List[int] = []
    complete = False
    k_min = round(num_ret_points * (1 - tolerance))
    k_max = round(num_ret_points * (1 + tolerance))

    while not complete:
        width = low + (high - low) / 2
        if width == prev_width or low > high:
            indices_list = result
            break

        c = width / 2.0
        num_cell_cols = int(math.floor(cols / c))
        num_cell_rows = int(math.floor(rows / c))
        covered_vec = [[False for _ in range(num_cell_cols + 1)] for _ in range(num_cell_rows + 1)]
        result = []

        for i in range(len(keypoints)):
            row = int(math.floor(keypoints[i].pt[1] / c))
            col = int(math.floor(keypoints[i].pt[0] / c))
            if not covered_vec[row][col]:
                result.append(i)
                row_min = int((row - math.floor(width / c)) if ((row - math.floor(width / c)) >= 0) else 0)
                row_max = int(
                    (row + math.floor(width / c)) if ((row + math.floor(width / c)) <= num_cell_rows) else num_cell_rows
                )
                col_min = int((col - math.floor(width / c)) if ((col - math.floor(width / c)) >= 0) else 0)
                col_max = int(
                    (col + math.floor(width / c)) if ((col + math.floor(width / c)) <= num_cell_cols) else num_cell_cols
                )
                for row_to_cover in range(row_min, row_max + 1):
                    for col_to_cover in range(col_min, col_max + 1):
                        if not covered_vec[row_to_cover][col_to_cover]:
                            covered_vec[row_to_cover][col_to_cover] = True

        if k_min <= len(result) <= k_max:
            indices_list = result
            complete = True
        elif len(result) < k_min:
            high = width - 1
        else:
            low = width + 1
        prev_width = width

    return indices_list
