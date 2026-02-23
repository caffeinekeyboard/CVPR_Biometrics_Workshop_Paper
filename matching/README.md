# Matching metrics

## Minutiae-based matching (mathematical description)

The `minutiae_metric` function compares two sets of minutiae assuming the fingerprints are already aligned. Each minutia is represented as a 3‑tuple $(x, y, \theta)$, where $(x, y)$ is the location and $\theta$ is the local ridge orientation.

Let
- $M_1 = \{(x_i, y_i, \theta_i)\}_{i=1}^{N_1}$ and
- $M_2 = \{(u_j, v_j, \phi_j)\}_{j=1}^{N_2}$.

### 1) Candidate match test
A minutia pair $(i, j)$ is considered *compatible* if both its Euclidean distance and orientation difference are below thresholds:

$$
\lVert (x_i, y_i) - (u_j, v_j) \rVert_2 \le d_{\text{th}},
$$

$$
\Delta\theta_{ij} = \min\big(|\theta_i-\phi_j|, 2\pi- |\theta_i-\phi_j|\big) \le \theta_{\text{th}}.
$$

### 2) Greedy one‑to‑one matching
Among all compatible pairs, a greedy procedure enforces one‑to‑one matches: for each minutia in $M_1$, choose the closest unmatched compatible minutia in $M_2$. This yields a match count $C$.

### 3) Symmetric match ratio
The match ratio is

$$
R = \frac{2C}{N_1 + N_2}.
$$

### 4) Score with minimum ratio threshold
The final score is

$$
S =
\begin{cases}
0, & R < r_{\text{th}},\\
\min(1, R), & R \ge r_{\text{th}}.
\end{cases}
$$

where $d_{\text{th}}$ is the distance threshold, $\theta_{\text{th}}$ is the angle threshold, and $r_{\text{th}}$ is the minimum match ratio threshold.
