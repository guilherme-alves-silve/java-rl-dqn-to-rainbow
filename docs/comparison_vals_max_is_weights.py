"""
author: Guilherme Alves Silveira
incremented with AI tools
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def importance_sampling_weights(sampling_prob: list, beta: float = 0.4):
    """
    w_i = (N * P(i))^{-beta}
    max_j w_j <- We'll investigate this max operation
    w_norm_i = w_i/ max_j w_j
    Let's remember that w_norm_i must have minimum value of 0 and a maximum of 1, range [0, 1].
    And, as we divide by a value less them zero, the less the value, the bigger is the result,
    where n > 0.
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
    :param sampling_prob:
    :return:
    """
    n = len(sampling_prob)
    p_sum = sum(sampling_prob)
    p_min = min(sampling_prob)/p_sum

    max_is_weight = (n * p_min)**(-beta)

    p_samples = [p_sample/p_sum for p_sample in sampling_prob]

    is_weights = [(n * p_sample)**(-beta) for p_sample in p_samples]
    norm_is_weights = [is_weight/max_is_weight for is_weight in is_weights]
    return np.array(norm_is_weights)

sampling_prob = [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005]
sampling_prob.sort(reverse=True)

result = importance_sampling_weights(sampling_prob)
print("sampling_prob:", sampling_prob)
print("result:", result)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(result))
colors = cm.Blues(0.2 + 0.8 * result)

bars = ax.bar(x, result, color=colors)
ax.set_xticks(x)
ax.set_xticklabels([f'{p:.5f}' for p in sampling_prob], rotation=45, ha='right')
ax.bar_label(bars, fmt='%.2f', padding=1)
ax.set_title("The smaller the value of $P_{\\min}$ the grater the greater $w_{norm}$ will be (inversely proportional)")
ax.set_xlabel("$P(i)$")
ax.set_ylabel("$\\frac{w_i}{\\max_j(w_j)}$")
plt.grid()

plt.savefig('graphics/comparison_vals_max_is_weights.jpg', dpi=150, bbox_inches='tight')
plt.show()
