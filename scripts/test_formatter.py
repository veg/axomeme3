from Bio import Phylo
from io import StringIO
import pyvolve

def to_newick(node, target_names_to_flag, flag_str="#m1"):
    name_str = node.name if node.name else ""
    bl_str = f":{node.branch_length:.6f}" if node.branch_length is not None else ""
    if node.name and node.name in target_names_to_flag:
        bl_str = f"{bl_str}{flag_str}"
    if node.is_terminal():
        return f"{name_str}{bl_str}"
    else:
        children_str = ",".join(to_newick(child, target_names_to_flag, flag_str) for child in node.clades)
        return f"({children_str}){name_str}{bl_str}"

# Test with a simple tree
tree_str = "((sp1:0.1,sp2:0.1)node1:0.25,sp3:0.3);"
tree = Phylo.read(StringIO(tree_str), "newick")

# Name internal nodes if they don't have names
for i, clade in enumerate(tree.get_nonterminals()):
    if not clade.name:
        clade.name = f"intnode_{i}"

print("Original terminals:", [n.name for n in tree.get_terminals()])
print("Original nonterminals:", [n.name for n in tree.get_nonterminals()])

# Flag sp1 and node1 (intnode_1)
formatted_tree = to_newick(tree.clade, ["sp1", "intnode_1"], "#m1") + ";"
print("\nFormatted tree string:")
print(formatted_tree)

# Verify Pyvolve can parse it
try:
    my_pyvolve_tree = pyvolve.read_tree(tree=formatted_tree)
    print("\nPyvolve successfully parsed the tree!")
except Exception as e:
    print(f"\nPyvolve parsing failed: {e}")
