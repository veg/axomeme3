import pyvolve

# Set up a small tree with branch model flags for foregrounds only
tree_str = "((sp1:0.1#m1,sp2:0.1#m2)node1:0.1,sp3:0.2);"
my_tree = pyvolve.read_tree(tree=tree_str)

try:
    print("Testing branch-heterogeneous model initialization with Partition...")
    m1 = pyvolve.Model("codon", {"omega": 4.0, "kappa": 2.0}, name="m1")
    m2 = pyvolve.Model("codon", {"omega": 2.0, "kappa": 2.0}, name="m2")
    rootmodel = pyvolve.Model("codon", {"omega": 0.1, "kappa": 2.0}, name="rootmodel")
    
    # Try partition and evolver
    part = pyvolve.Partition(models=[m1, m2, rootmodel], size=10, root_model_name="rootmodel")
    evolver = pyvolve.Evolver(tree=my_tree, partitions=part)
    evolver(seqfile="test_branch_het_2.fasta", write_joint_states=False)
    print("Success! Simulated sequence.")
except Exception as e:
    print(f"Failed: {e}")
