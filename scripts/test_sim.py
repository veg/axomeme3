import pyvolve

def make_balanced_tree(n_leaves, branch_length=0.05):
    leaves = [f"sp{i}" for i in range(1, n_leaves + 1)]
    while len(leaves) > 1:
        next_level = []
        for i in range(0, len(leaves), 2):
            next_level.append(f"({leaves[i]}:{branch_length},{leaves[i+1]}:{branch_length})")
        leaves = next_level
    return leaves[0] + ";"

tree_str = make_balanced_tree(64, 0.05)
my_tree = pyvolve.read_tree(tree=tree_str)

# Test codon model parameters
try:
    print("Testing 'omega' parameter...")
    model_1 = pyvolve.Model("codon", {"omega": 0.1, "kappa": 2.0})
    print("Success with 'omega'!")
except Exception as e:
    print(f"Failed with 'omega': {e}")
    try:
        print("Testing 'dN/dS' parameter...")
        model_1 = pyvolve.Model("codon", {"dN/dS": 0.1, "kappa": 2.0})
        print("Success with 'dN/dS'!")
    except Exception as e2:
        print(f"Failed with 'dN/dS': {e2}")
