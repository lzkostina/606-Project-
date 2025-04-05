import numpy as np
import matplotlib.pyplot as plt

def mask_entries(image, sparsity_level=0.3,fill_value=0.0):
    mask = np.random.rand(*image.shape) < sparsity_level
    masked_image = image.copy()
    masked_image[mask] = fill_value
    return masked_image, mask

def calculate_mse(original, predicted, mask=True):
    error = original[mask] - predicted[mask]
    mse = np.mean(error ** 2)
    return mse

def calculate_psnr(original, predicted, mask=True, max_pixel=1.0):
    error = original[mask] - predicted[mask]
    mse = np.mean(error ** 2)
    if mse == 0:
        return float('inf')  
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def pmf_matrix_completion(image_PMF, mask, k=50, sigma2=0.001, lambda_U=0.1, lambda_V=0.1,
                           learning_rate=0.01, num_epochs=300, verbose=True):
    m, n = image_PMF.shape
    U = np.random.normal(scale=0.1 / np.sqrt(lambda_U), size=(m, k))
    V = np.random.normal(scale=1.0 / np.sqrt(lambda_V), size=(n, k))
    observed_pixels = np.array([(i, j) for i in range(m) for j in range(n) if not mask[i, j]])

    for epoch in range(num_epochs):
        np.random.shuffle(observed_pixels)
        total_error = 0

        for i, j in observed_pixels:
            pred = np.dot(U[i], V[j])
            error = image_PMF[i, j] - pred

            grad_U = ((error * V[j]) / sigma2 - lambda_U * U[i])
            grad_V = ((error * U[i]) / sigma2 - lambda_V * V[j])

            grad_U = np.clip(grad_U, -100, 100)
            grad_V = np.clip(grad_V, -100, 100)

            U[i] += learning_rate * grad_U
            V[j] += learning_rate * grad_V
            total_error += error ** 2

    return U, V