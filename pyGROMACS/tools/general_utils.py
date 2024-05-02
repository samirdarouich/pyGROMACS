import numpy as np

def generate_series(desired_mean, desired_std, size):
    # Generate random numbers from a standard normal distribution
    random_numbers = np.random.randn(size)
    
    # Calculate the Z-scores
    z_scores = (random_numbers - np.mean(random_numbers)) / np.std(random_numbers)
    
    # Scale by the desired standard deviation and shift by the desired mean
    series = z_scores * desired_std + desired_mean
    
    return series