import numpy as np

def vectorize(A, B):
    return np.concatenate([A.flatten(), B.flatten()])

def unvectorize(x, m, n, r):
    A = x[:m * r].reshape(m, r)
    B = x[m * r:].reshape(n, r)
    return A, B

def loss_fn(A, B, M, W, lambda1, lambda2):
    residual = W * (M - A @ B.T)
    return np.sum(residual**2) + lambda1 * np.sum(A**2) + lambda2 * np.sum(B**2)

def compute_gradient(A, B, M, W, lambda1, lambda2):
    #M: m*n matrix
    m, r = A.shape
    n = B.shape[0]
    grad_A = np.zeros_like(A)
    grad_B = np.zeros_like(B)
    for i in range(m):
        wi = W[i, :]
        mi = M[i, :]
        diag_wi = np.diag(wi)
        grad_A[i] = 2 * B.T @ diag_wi @ (B @ A[i] - mi) + 2 * lambda1 * A[i]
    for j in range(n):
        wj = W[:, j]
        mj = M[:, j]
        diag_wj = np.diag(wj)
        grad_B[j] = 2 * A.T @ diag_wj @ (A @ B[j] - mj) + 2 * lambda2 * B[j]
    return np.concatenate([grad_A.flatten(), grad_B.flatten()])

def compute_hessian(A, B, W, lambda1, lambda2):
    m, r = A.shape
    n = B.shape[0]
    H_size = (m + n) * r
    H = np.zeros((H_size, H_size))
    for i in range(m):
        wi = np.diag(W[i, :])
        H_aa = 2 * B.T @ wi @ B + 2 * lambda1 * np.eye(r)
        idx = slice(i * r, (i + 1) * r)
        H[idx, idx] = H_aa
    for j in range(n):
        wj = np.diag(W[:, j])
        H_bb = 2 * A.T @ wj @ A + 2 * lambda2 * np.eye(r)
        idx = slice((m + j) * r, (m + j + 1) * r)
        H[idx, idx] = H_bb
    return H

def damped_newton(M, W, r, lambda1=1e-3, lambda2=1e-3, max_iter=100, tol=1e-6):
    m, n = M.shape
    A = np.random.randn(m, r)
    B = np.random.randn(n, r)
    x = vectorize(A, B)
    lamb = 0.01

    for it in range(max_iter):
        A, B = unvectorize(x, m, n, r)
        grad = compute_gradient(A, B, M, W, lambda1, lambda2)
        H = compute_hessian(A, B, W, lambda1, lambda2)

        while True:
            try:
                H_reg = H + lamb * np.eye(H.shape[0])
                dx = np.linalg.solve(H_reg, grad)
                x_new = x - dx
                A_new, B_new = unvectorize(x_new, m, n, r)
                F_old = loss_fn(A, B, M, W, lambda1, lambda2)
                F_new = loss_fn(A_new, B_new, M, W, lambda1, lambda2)
                if F_new < F_old:
                    x = x_new
                    lamb *= 0.1
                    break
                else:
                    lamb *= 10
            except np.linalg.LinAlgError:
                lamb *= 10

        if np.linalg.norm(dx) < tol:
            break

    return unvectorize(x, m, n, r)

def damped_newton_linesearch(M, W, r, lambda1=1e-3, lambda2=1e-3, max_iter=100, tol=1e-6):
    m, n = M.shape
    A = np.random.randn(m, r)
    B = np.random.randn(n, r)
    x = vectorize(A, B)
    lamb = 1e-2

    for it in range(max_iter):
        A, B = unvectorize(x, m, n, r)
        grad = compute_gradient(A, B, M, W, lambda1, lambda2)
        H = compute_hessian(A, B, W, lambda1, lambda2)

        try:
            H_reg = H + lamb * np.eye(H.shape[0])
            dx = -np.linalg.solve(H_reg, grad)
        except np.linalg.LinAlgError:
            lamb *= 10
            continue

        # Line search
        alpha = 1.0
        F0 = loss_fn(A, B, M, W, lambda1, lambda2)
        for _ in range(20):
            x_new = x + alpha * dx
            A_new, B_new = unvectorize(x_new, m, n, r)
            F_new = loss_fn(A_new, B_new, M, W, lambda1, lambda2)
            if F_new < F0:
                break
            alpha *= 0.5

        x = x + alpha * dx
        if np.linalg.norm(alpha * dx) < tol:
            break

        # Adjust lambda: shrink if step was large
        if alpha > 0.9:
            lamb *= 0.1
        elif alpha < 0.1:
            lamb *= 10

    return unvectorize(x, m, n, r)