import random
import numpy as np
from scipy import stats

'''
if alpha = beta = 1 then we have uniform distribution
if alpha = beta     then we have a symmetric distribution, where x = 1/2
if alpha > beta     then the density is right-leaning (concentrated in the neighbourhood of 1)
                    the mean and variance will be able to be computed explicitly
'''


def thompson_rule_update(alpha: int = 1, beta: int = 1) -> int:

    pass
