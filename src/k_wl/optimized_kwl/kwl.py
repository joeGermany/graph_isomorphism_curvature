from pathlib import Path
import ast
import subprocess
from pathlib import Path

K_WL_DIR=Path(__file__).resolve().parent.parent/"k_wl"

if (K_WL_DIR/"run_wl").exists():
    RUN_WL_EXE=K_WL_DIR/"run_wl"
elif (K_WL_DIR/"run_wl.exe").exists():
    RUN_WL_EXE=K_WL_DIR/"run_wl.exe"
else:
    raise FileNotFoundError("Could not find run_wl executable.")

def write_graph(G,filename):
    edges=sorted((min(int(u),int(v)),max(int(u),int(v))) for u,v in G.edges())

    with open(filename,"w",encoding="utf-8") as f:
        f.write(f"{G.number_of_nodes()} {len(edges)}\n")

        for u,v in edges:
            f.write(f"{u} {v}\n")


def write_partition(filename,classes):
    classes=list(classes.values()) if isinstance(classes,dict) else classes
    classes=[sorted(map(int,cls)) for cls in classes if cls]

    with open(filename,"w",encoding="utf-8") as f:
        f.write(f"{len(classes)}\n")

        for cls in classes:
            f.write(f"{cls}\n")


def read_partition(filename):
    with open(filename,"r",encoding="utf-8") as f:
        number_of_classes=int(next(f))
        return [sorted(ast.literal_eval(next(f))) for _ in range(number_of_classes)]


def run_kwl(G,k,initial_partition=None,output_dir="kwl_output"):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)

    graph_file=output_dir/"graph.txt"
    partition_file=output_dir/"input_partition.txt"
    output_file=output_dir/f"{k}wl_partition.txt"

    write_graph(G,graph_file)

    if initial_partition is None:
        partition_argument="-"
    else:
        write_partition(partition_file,initial_partition)
        partition_argument=str(partition_file)

    command=[str(RUN_WL_EXE),str(graph_file),str(k),partition_argument,str(output_file)]
    result=subprocess.run(command,text=True,capture_output=True)

    if result.returncode!=0:
        raise RuntimeError(result.stderr)

    return read_partition(output_file)