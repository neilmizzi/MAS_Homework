from random import random
import numpy as np

"""
if alpha = beta = 1 then we have uniform distribution
if alpha = beta     then we have a symmetric distribution, where x = 1/2
if alpha > beta     then the density is right-leaning (concentrated in the neighbourhood of 1)
                    the mean and variance will be able to be computed explicitly
"""


# Gets a Beta Distribution and applies the Thompson Update Rule
def thompson_update_rule(alpha: int = 1, beta: int = 1) -> (int, int):
    sample = np.random.beta(alpha, beta)    # Generate Probability
    r = bandit_sample(sample)               # Get reward
    return alpha+r, beta+(1-r)              # Amend Alpha and Beta values accordingly


# returns reward depending on sample passed
def bandit_sample(p: float) -> int:
    r = random.random()
    return 1 if p > r else 0

