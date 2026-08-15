import networkx as nx
from sage.all import *
# from sage.graphs.strongly_regular_db import SRG_144_39_6_12
from sage.graphs.strongly_regular_db import SRG_175_72_20_36,SRG_176_105_68_54,SRG_176_49_12_14,SRG_176_90_38_54,SRG_196_91_42_42,SRG_210_99_48_45,SRG_220_84_38_28,SRG_243_110_37_60,SRG_253_140_87_65,SRG_276_140_58_84,SRG_280_117_44_52,SRG_280_135_70_60,SRG_416_100_36_20,SRG_560_208_72_80,SRG_630_85_20_10
# from sage.graphs.strongly_regular_db import SRG_from_RSHCD

# G = SRG_from_RSHCD(Integer(144), Integer(65), Integer(28), Integer(30))
# a = G.automorphism_group()
# print(len(a.orbits()))

arr = [SRG_175_72_20_36,SRG_176_105_68_54,SRG_176_49_12_14,SRG_176_90_38_54,SRG_196_91_42_42,SRG_210_99_48_45,SRG_220_84_38_28,SRG_243_110_37_60,SRG_253_140_87_65,SRG_276_140_58_84,SRG_280_117_44_52,SRG_280_135_70_60,SRG_416_100_36_20,SRG_560_208_72_80,SRG_630_85_20_10]
names = []
for i in range(len(arr)):
    G = arr[i]()
    param = G.is_strongly_regular(parameters=True)
    param = list(param)
    name = f"{param[0]}-{param[1]}-{param[2]}-{param[3]}"
    names.append(name)
    print(names)

    graph = G.networkx_graph()
    graph = nx.convert_node_labels_to_integers(graph)

    filename = f"graphs/sage/{name}"

    with open(filename, "w") as f:
        f.write(f"c {filename}\n")
        f.write(
            f"p edge {graph.number_of_nodes()} "
            f"{graph.number_of_edges()}\n"
        )

        # NetworkX uses vertices 0,...,n-1
        # DIMACS uses vertices 1,...,n
        for u, v in graph.edges():
            f.write(f"e {u + 1} {v + 1}\n")