from functools import reduce
from typing import List, Optional, Set
import pandas as pd
import numpy as np
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.sparse import dok_matrix, csr_matrix
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from networkx.drawing.nx_agraph import graphviz_layout

from simulation.constants import *
from simulation.google_cloud import GoogleCloud

CONF = {"hops": False, "groups": True, "upload": False}
gcloud = GoogleCloud()


def show_graph(G, labels=False, partition=False):
    plt.figure(1, figsize=(8, 8))
    pos = graphviz_layout(G, prog="neato")  # nx.spring_layout(G)
    if partition:
        pass
        # cmap = cm.get_cmap("cool", max(partition.values()) + 1)
        # nc = list(partition.values())
        # nx.draw(G, pos, node_size=1500, cmap=cmap, node_color=nc, with_labels=True)
    else:
        nx.draw(G, pos, node_size=40, vmin=0.0, vmax=1.0, with_labels=True)
        try:
            if labels:
                el = {t[:2]: int(t[2]["hops"]) for t in G.edges.data()}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=el)
        except KeyError:
            pass
    plt.savefig(f"{'-'.join(list(map(str,shortest_path)))}.png")


def preview(df, time: str) -> pd.DataFrame:
    G = nx.from_pandas_edgelist(
        df[df["datetime"] == pd.to_datetime(time)],
        target="destination",
        edge_attr=True,
    )
    show_graph(G, labels=True)


def load() -> pd.DataFrame():
    print("loaded df")
    return pd.read_csv(
        DATA_FOLDER / "copenhagen_interactions.csv", parse_dates=["datetime"]
    ).head(158)


def hops(df):
    five_minutes = [snapshot for _, snapshot in df.resample("5T", on="datetime")]
    hops = []
    for fm in five_minutes:
        row, col = fm[["source", "destination"]].values.T
        data = [1] * len(row)
        size = max(np.append(row, col)) + 1
        # singletons = len([x for x in range(size) if x not in np.append(row,col)])
        graph = csr_matrix(
            (data * 2, (np.append(row, col), np.append(col, row))), shape=(size, size)
        )
        _, labels = connected_components(graph, directed=False)
        # cc -= singletons
        group_indices = [k for k, v in Counter(labels).items() if v > 2]
        group_participants = [i for i, e in enumerate(labels) if e in group_indices]
        sp = shortest_path(graph, directed=False, method="D")
        sp[sp == np.inf] = 0.0
        df = (
            pd.DataFrame({"hops": dict(dok_matrix(np.triu(sp)))})
            .rename_axis(["source", "destination"])
            .reset_index()
        )
        df["datetime"] = fm["datetime"].iloc[0]
        df["is_group"] = df["source"].isin(group_participants)
        hops.append(df)
    print("added hops")
    return pd.concat(hops)


def agg(df: pd.DataFrame) -> pd.DataFrame:
    df["meeting_nodes"] = df[["source", "destination"]].apply(
        lambda x: tuple(sorted(x)), axis=1
    )
    df = df.sort_values(["meeting_nodes", "datetime"]).reset_index(drop=True)
    df["meeting_id"] = (df["datetime"].diff() != pd.Timedelta("5m")).cumsum()
    cols = ["meeting_id", "meeting_nodes"] + (
        ["hops", "is_group"] if CONF["hops"] else []
    )
    df = (
        df.groupby(cols)
        .agg(
            **{
                "duration": pd.NamedAgg(
                    column="datetime", aggfunc=lambda x: x.count() * 5
                ),
                "datetime": pd.NamedAgg(column="datetime", aggfunc="min"),
            }
        )
        .reset_index()
    )
    df[["source", "destination"]] = pd.DataFrame(df["meeting_nodes"].tolist())
    df = df.sort_values("datetime")
    if not CONF["groups"]:
        df = df.drop(columns=["meeting_nodes", "meeting_id"])
    print("aggeregated durations")
    return df


def group_sets(a: List[Optional[Set[int]]], b: Set[int]):
    # FIXME: creates not necessarily full group
    for i, x in enumerate(a):
        if len(x & b) > 0:
            a[i] = x | b
            return a
    else:
        return a + [b]


def groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.groupby(["datetime", "duration"])["meeting_nodes"]
    # df = df.apply(list)
    df = df.apply(lambda x: [set(i) for i in x])
    df = df.apply(lambda x: list(reduce(group_sets, x, [])))
    df = df.explode()
    df = df.to_frame("group")
    df = df.reset_index()
    print("grouped meetings")
    return df


def export(df: pd.DataFrame):
    g = "_groups" if CONF["groups"] else ""
    h = "_hops" if CONF["hops"] else ""
    path = DATA_FOLDER / f"copenhagen_agg{g}{h}.csv"
    df.to_csv(path, index=False)
    print("exported to csv")
    if CONF["upload"]:
        gcloud.upload(path, bucket_name="simulation_datasets")
        print("uploaded to google cloud")


if __name__ == "__main__":
    df = load()
    if CONF["hops"]:
        df = hops(df)
    df = agg(df)
    if CONF["groups"]:
        df = groups(df)
    export(df)
