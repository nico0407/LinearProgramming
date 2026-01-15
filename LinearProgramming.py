import numpy as np
from scipy.optimize import linprog


def min_time_liquidation(A, f, x):
    """
    Solve: max(alpha)  s.t.  A^T u = alpha x.
    Bounds: -f_i < u_i < f_i   &   alpha >= 0.

    Returns: T_star, alpha_star, u_star 
    """

    A = np.asarray(A, float)
    f = np.asarray(f, float)
    x = np.asarray(x, float)
    m, n = A.shape

    # Decision variables: u (dimension m) and alpha (dimension 1) total vector of dimension m+1
    c = np.zeros(m + 1)
    c[-1] = -1.0  # maximize alpha, so minimize "-alpha"

    # Equality: A^T u - alpha x = 0
    Aeq = np.hstack([A.T, -x.reshape(-1, 1)])
    beq = np.zeros(n)

    # Bounds
    bounds = [(-fi, fi) for fi in f] + [(0, None)]

    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")

    if not res.success:
        return np.inf, 0.0, None, res

    u_star = res.x[:m]
    alpha_star = res.x[-1]

    if alpha_star <= 0:
        return np.inf, alpha_star, u_star

    T_star = 1.0 / alpha_star
    return T_star, alpha_star, u_star



A = np.array([
    [-1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0, -1],
    [ 1, -1,  0],
    [ 1,  0, -1],
    [ 0,  1, -1],
], float)

f = np.array([1000, 100, 100, 10, 10, 20333], float)  # per minute
x = np.array([12345, -20000, 340000], float)

T_star, alpha_star, u_star = min_time_liquidation(A, f, x)

print(f"alpha* = {alpha_star:.8f} per minute")
print(f"T*    = {T_star:.4f} minutes ({T_star/60:.2f} hours)")
print("u* per minute =", u_star)

# Sanity check A^T u = alpha x
print("A^T u* =", A.T @ u_star)
print("alpha* x =", alpha_star * x)
