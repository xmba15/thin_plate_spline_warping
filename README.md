# 📝 2D Thin Plate Spline Warping: Implementations & Experiments

---

Simple samples to demonstrate 2D Thin Plate Spline Warping.

The keypoint matches are obtained by lightglue matcher on KeyNetAffNetHardNet keypoints.

## :book: Explanation

---

2D Thin Plate Spline (TPS) warping is a parametric transformation method to transform a source image to a target image.

TPS mapping function $`\textbf{f}(\textbf{p})=\textbf{f}(x,y)`$ is estimated using control points with source points $`{\left\{ \textbf{p}_i=(x_i,y_i) \right\}}_{i=1}^N`$ in the source image that match with target points $`{\left\{ \textbf{p'}_i=(x'_i,y'_i) \right\}}_{i=1}^N`$.

The loss function in 2D TPS warping is composed of two main components: data term and bending energy

$$
L=\sum_{i=1}^{N}\left\| f(\textbf{p}_i)-\textbf{p}'_i \right\|^2+\lambda\int\int\left[ \left( \frac{\partial ^2 f}{\partial x^2} \right)^2+2\left( \frac{\partial ^2f}{\partial x\partial y} \right)^2+\left(\frac{\partial ^2f}{\partial y^2}  \right)^2 \right]dxdy
$$

This optimization problem has a closed form solution as following:

$$
\textbf{f}(\textbf{p})=(f_x(x,y),f_y(x,y))
$$

$$
f_x(x,y)=a_1+a_2x+a_3y+\sum_{i=1}^{N}w_iU(\left\| \textbf{p}-\textbf{p}_i \right\|)
$$

$$
f_y(x,y)=b_1+b_2x+b_3y+\sum_{i=1}^{N}v_iU(\left\| \textbf{p}-\textbf{p}_i \right\|)
$$

where $`U(r)=r^2log(r^2)`$ is radial basis function, with $`r=\left\| \textbf{p}-\textbf{p}_i \right\|`$

This system of equations has the following constraint:

$$
\sum_{i=1}^{N}w_i=\sum_{i=1}^{N}w_ix_i=\sum_{i=1}^{N}w_iy_i=0
$$

$$
\sum_{i=1}^{N}v_i=\sum_{i=1}^{N}v_ix_i=\sum_{i=1}^{N}v_iy_i=0
$$

Hence, can be transformed to the matrix formulation $`\textbf{A}\textbf{c}=\textbf{b}`$

$$
A=\begin{bmatrix}
\textbf{K} & \textbf{P}\\
\textbf{P}^T & \textbf{0}
\end{bmatrix}
,
c=\begin{bmatrix}
\textbf{w} \\
\textbf{a}
\end{bmatrix}
,
b=\begin{bmatrix}
\textbf{x}' \\
\textbf{0}
\end{bmatrix}
$$

$$
K_{ij}=U(\left\| \textbf{p}_i-\textbf{p}_j \right\|);\textbf{K}\in \textbf{R}^{N \times N}
$$

$$
\textbf{P}=\begin{bmatrix}
1 & x_1 & y_1 \\
1 & x_2 & y_2 \\
... & ... & ... \\
1 & x_N & y_N
\end{bmatrix}; \textbf{P} \in \textbf{R}^{N \times 3}
$$

$$
\textbf{w}=[w_1w_2...w_N]^T;\textbf{a}=[a_1a_2a_3]^T;\textbf{x}'=[x_1'x_2'...x_N']^T
$$

## :running: How to Run

---

```
python3 scripts/test_tps.py --query_path ./data/car1.jpg --ref_path ./data/car2.jpg
```

<table>
  <tr>
    <td align="center">
      <img src="./data/car1.jpg" alt="car1" width="45%" /><br />
      <b>query image</b>
    </td>
    <td align="center">
      <img src="./data/car2.jpg" alt="car2" width="45%" /><br />
      <b>reference image</b>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <img src="./docs/images/warped_image.jpg" alt="car1" width="45%" /><br />
      <b>warped query image</b>
    </td>
    <td align="center">
      <img src="./docs/images/merged_image.jpg" alt="car2" width="45%" /><br />
      <b>warped query image blended with reference</b>
    </td>
  </tr>
</table>

## 🎛 Development environment

---

```bash
mamba env create --file environment.yml
mamba activate tps
```

## :gem: References

---

- [Thin Plate Spline](https://en.wikipedia.org/wiki/Thin_plate_spline)
