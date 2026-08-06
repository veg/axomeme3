import pyvolve

# Set up a small tree
tree_str = "((sp1:0.1,sp2:0.1)node1:0.1,sp3:0.2);"
my_tree = pyvolve.read_tree(tree=tree_str)

try:
    print("Testing branch-heterogeneous model initialization...")
    # Dictionary with branch names as keys and a default key
    model = pyvolve.Model("codon", {"omega": {"sp1": 4.0, "node1": 2.0, "default": 0.1}})
    print("Success! Model initialized.")
    
    # Try partition and evolver
    part = pyvolve.Partition(models=model, size=10)
    evolver = pyvolve.Evolver(tree=my_tree, partitions=part)
    evolver(seqfile="test_branch_het.fasta", write_joint_states=False)
    print("Success! Simulated sequence.")
except Exception as e:
    print(f"Failed: {e}")
