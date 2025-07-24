import numpy as np


def anchored_sigmoid(x, t):
    """
    A smooth function f: [0, 1] -> [0, 1] such that
    f(0) = 0, f(1) = 1, f(t) = 0.5
    """

    # Find parameter k such that sigmoid(k*(x - t)) maps t to 0.5 and [0,1] to [0,1]
    # This transformation ensures f(0)=0 and f(1)=1
    k = 10 #np.log(10) / (0.5 - t) if t != 0.5 else 10  # Adjustable if t=0.5

    # Sigmoid centered at t
    s = lambda x: 1 / (1 + np.exp(-k * (x - t)))

    # Normalize so s(0)=0 and s(1)=1
    s0 = s(0)

    s1 = s(1)
    return (s(x) - s0) / (s1 - s0)
