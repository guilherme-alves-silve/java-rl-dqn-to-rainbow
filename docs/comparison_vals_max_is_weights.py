"""
author: Guilherme Alves Silveira
incremented with AI tools
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def importance_sampling_weights(sampling_prob_param: list,
                                beta: float = 0.4) -> np.ndarray:
    """
    w_i = (N * P(i))^{-beta}
    max_j w_j <- We'll investigate this max operation
    w_norm_i = w_i/ max_j w_j
    Let's remember that w_norm_i must range of (0, 1], and as we divide by a value less them one,
    the less the value, the bigger is the result, where value > 0
    10 = 1/0.1
    20 = 1/0.05
    100 = 1/0.01
    200 = 1/0.005
    1000 = 1/0.001
    ...
    Other observations: All values are first normalized to be in range [0, 1], because
    it's a probability, then we calculate IS. It avoids value overflow, good practice to do that.

    Based on:
        https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/03_per.py#L345
    :param beta: Force of IS normalization weights
    :param sampling_prob_param: Sampling probability (returned from array O(n)
        or O(log N) when Segment Tree is used
    :return: np.ndarray
    """
    n = len(sampling_prob_param)
    p_min = min(sampling_prob_param)
    p_sum = sum(sampling_prob_param)

    max_is_weight = (n * p_min/p_sum)**(-beta)
    p_samples = [p_sample / p_sum for p_sample in sampling_prob_param]
    is_weights = [(n * p_sample)**(-beta) for p_sample in p_samples]
    norm_is_weights = [is_weight/max_is_weight for is_weight in is_weights]
    return np.array(norm_is_weights)


def fmt_prob(p_min: float, p_max: float, p: float) -> str:
    if np.isclose(p, p_max, rtol=1e-5, atol=1e-8):
        return f'{p:.1f} = $P_{{\\max}}$'
    if np.isclose(p, p_min, rtol=1e-5, atol=1e-8):
        return f'{p:.5f} = $P_{{\\min}}$'
    return f'{p:.6f}'.rstrip('0').rstrip('.')


sampling_prob = [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005]
sampling_prob.sort()
p_min = min(sampling_prob)
p_max = max(sampling_prob)

result = importance_sampling_weights(sampling_prob)
print("sampling_prob:", sampling_prob)
print("result:", result)

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(result))
colors = cm.Blues(0.2 + 0.8 * result)

bars = ax.bar(x, result, color=colors)
ax.set_xticks(x)
ax.set_xticklabels([fmt_prob(p_min, p_max, p) for p in sampling_prob], rotation=45, ha='right')
ax.bar_label(bars, fmt='%.2f', padding=1)
ax.set_title("The smaller the value of $P_{\\min}$ the greater $w_{norm}$ will be (inversely proportional)")
ax.set_xlabel("$P(i)$")
ax.set_ylabel("$\\frac{w_i}{\\max_j(w_j)}$")
plt.grid()

plt.savefig('graphics/comparison_vals_max_is_weights.jpg', dpi=150, bbox_inches='tight')
