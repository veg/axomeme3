import pyvolve
import numpy as np

# Try state_freqs of size 61
try:
    print("Testing 'state_freqs' size 61...")
    freqs = np.ones(61) / 61.0
    model = pyvolve.Model("mutsel", {"state_freqs": freqs.tolist()})
    print("Success with state_freqs!")
except Exception as e:
    print(f"Failed with state_freqs: {e}")

# Try fitness of size 61
try:
    print("\nTesting 'fitness' size 61...")
    fitness = np.zeros(61)
    model = pyvolve.Model("mutsel", {"fitness": fitness.tolist()})
    print("Success with fitness size 61!")
except Exception as e:
    print(f"Failed with fitness size 61: {e}")

# Try fitness of size 20 (for amino acids)
try:
    print("\nTesting 'fitness' size 20...")
    fitness = np.zeros(20)
    model = pyvolve.Model("mutsel", {"fitness": fitness.tolist()})
    print("Success with fitness size 20!")
except Exception as e:
    print(f"Failed with fitness size 20: {e}")
