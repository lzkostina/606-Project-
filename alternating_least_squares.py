import numpy as np

def als(masked_image, mask, num_factors=20, lambda_reg=0.01, num_epochs=1000):
    num_users, num_items = masked_image.shape
    
    # Initialize U and V with random values (mean 0 and st. dev. 0.1 so that latent factors can both take positive and negative values, symmetry helps)
    U = np.random.normal(0, 0.1, (num_users, num_factors))
    V = np.random.normal(0, 0.1, (num_items, num_factors))
    
    # Perform ALS for _ epochs 
    for epoch in range(num_epochs):
        # Fix V and update U
        for i in range(num_users):
            V_masked = V[~mask[i, :]] 
            R_masked = masked_image[i, ~mask[i, :]]
            if V_masked.shape[0] > 0:  # make sure there are unmasked entries first so there aren't errors
                U[i, :] = np.linalg.solve(np.dot(V_masked.T, V_masked) + lambda_reg * np.eye(num_factors),
                                          np.dot(V_masked.T, R_masked))
        
        # Fix U and update V
        for j in range(num_items):
            U_masked = U[~mask[:, j], :] 
            R_masked = masked_image[~mask[:, j], j]
            if U_masked.shape[0] > 0:  # make sure there are unmasked entries first so there aren't errors
                V[j, :] = np.linalg.solve(np.dot(U_masked.T, U_masked) + lambda_reg * np.eye(num_factors),
                                          np.dot(U_masked.T, R_masked))
        '''
        # Print the loss so we can see how it changes throughout the epochs
        if (epoch + 1) % 10 == 0:
            loss = 0
            for i in range(num_users):
                for j in range(num_items):
                    if not mask[i, j]:
                        prediction = np.dot(U[i, :], V[j, :].T)
                        error = masked_image[i, j] - prediction
                        loss += error**2
                        
            # The contribution by regularization
            loss += lambda_reg * (np.sum(U**2) + np.sum(V**2))
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
        '''
    
    # Reconstructed image
    R_pred = np.dot(U, V.T)
    return R_pred
