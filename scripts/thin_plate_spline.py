import warnings

import numpy as np

__all__ = [
    "ThinPlateSpline",
]


class ThinPlateSpline:
    def __init__(self):
        self._coeffs = None
        self._src_points = None

    def fit(
        self,
        src_points_arr: np.ndarray,
        target_points_arr: np.ndarray,
    ):
        assert len(src_points_arr) == len(target_points_arr)
        assert len(src_points_arr.shape) == 2 and src_points_arr.shape[1] == 2
        assert len(target_points_arr.shape) == 2 and target_points_arr.shape[1] == 2

        self._solve_tps_system(
            src_points_arr,
            target_points_arr,
        )
        self._src_points = src_points_arr

    def predict(self, points_arr: np.ndarray):
        num_points = self._src_points.shape[0]
        a = self._coeffs[num_points:]
        w = self._coeffs[:num_points]

        # the more points, the more cpu usage will be needed for the following processing
        pairwise_dists = np.linalg.norm(self._src_points[:, None] - points_arr[None, :], axis=-1)

        _u_mat = self._rbf(pairwise_dists)
        del pairwise_dists

        non_affine_part = np.dot(_u_mat.T, w)
        del _u_mat

        affine_part = a[0] + np.dot(points_arr, a[1:])

        return affine_part + non_affine_part

    def _solve_tps_system(
        self,
        src_points_arr: np.ndarray,
        target_points_arr: np.ndarray,
    ):
        _k_mat, _p_mat = self._build_tps_matrices(src_points_arr)
        num_points = src_points_arr.shape[0]
        _l_mat = np.zeros((num_points + 3, num_points + 3))
        _l_mat[:num_points, :num_points] = _k_mat
        del _k_mat

        _l_mat[:num_points, num_points:] = _p_mat
        _l_mat[num_points:, :num_points] = _p_mat.T
        del _p_mat

        _y_mat = np.zeros((num_points + 3, 2))
        _y_mat[:num_points, :] = target_points_arr

        self._coeffs = np.linalg.solve(_l_mat, _y_mat)

    def _build_tps_matrices(
        self,
        src_points_arr,
    ):
        # num_points x num_points x 2
        diff = src_points_arr[:, None, :] - src_points_arr[None, :, :]
        pairwise_dists = np.linalg.norm(diff, axis=2)
        # kernel matrix
        _k_mat = self._rbf(pairwise_dists)

        # P matrix
        num_points = src_points_arr.shape[0]
        _p_mat = np.hstack((np.ones((num_points, 1)), src_points_arr))

        return _k_mat, _p_mat

    def _rbf(self, r):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.where(np.isclose(r, 0), 0, r**2 * np.log(r**2))
