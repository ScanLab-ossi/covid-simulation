from simulation.constants import *
import pandas as pd
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import networkx as nx
import argparse, sys
from yaml import Loader, load, Dumper, dump


def show_graph(G, labels=False):
    plt.figure(1, figsize=(8, 8))
    pos = graphviz_layout(G, prog="neato")
    nx.draw(G, pos, node_size=40, vmin=0.0, vmax=1.0, with_labels=True)
    try:
        if labels:
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels={t[:2]: int(t[2]["hops"]) for t in G.edges.data()}
            )
    except KeyError:
        pass
    plt.show()
    plt.savefig(DATA_FOLDER / "vis.png")


def to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df["datetime"] = pd.to_datetime("03/10/1995") + (
        (df["datetime"] * 5).astype(str) + "m"
    ).apply(pd.Timedelta)
    return df


def group_by_duration(df: pd.DataFrame) -> pd.DataFrame:
    df["meeting_nodes"] = df[["source", "destination"]].apply(
        lambda x: tuple(sorted(x)), axis=1
    )
    df = df.sort_values(["meeting_nodes", "datetime"]).reset_index(drop=True)
    df["meeting_id"] = (df["datetime"].diff() != pd.Timedelta("5m")).cumsum()
    df = (
        df.groupby(["meeting_id", "meeting_nodes"])
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
    source_and_dest = df["meeting_nodes"].tolist()
    df[["source", "destination"]] = source_and_dest
    df["group"] = ["{" + ", ".join(map(str, l)) + "}" for l in source_and_dest]
    df = df.drop(columns=["meeting_nodes", "meeting_id"]).sort_values("datetime")
    return df


def add_to_datasets(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp):
    with open(CONFIG_FOLDER / "datasets.yaml", "r") as f:
        d = load(f, Loader=Loader)
    print(df[["source", "destination"]].max().max())
    if not args.filename in d:
        d[args.filename] = {
            "storage": "csv",
            "nodes": int(df[["source", "destination"]].max().max()) + 1,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "interval": "day",
            "groups": args.groups,
            "hops": args.hops,
        }
    else:
        print("Didn't add to datasets.yaml - dataset already exists!")
    with open(CONFIG_FOLDER / "datasets.yaml", "w") as f:
        dump(d, f, Dumper=Dumper)


def parse_args(args):
    parser = argparse.ArgumentParser(description="Let's crunch some data!")
    parser.add_argument("filename", type=str, help="What is the raw data's filename?")
    parser.add_argument(
        "--snapshot",
        action=argparse.BooleanOptionalAction,
        help="Should I print a graphical representation of the first timestamp?",
        default=False,
    )
    parser.add_argument(
        "--dataset-metadata",
        action=argparse.BooleanOptionalAction,
        help="Should I add the metadata to the datasets.yaml?",
        default=False,
    )
    parser.add_argument(
        "--groups",
        action=argparse.BooleanOptionalAction,
        help="Should source and destination be grouped?",
        default=True,
    )
    parser.add_argument(
        "--hops",
        action=argparse.BooleanOptionalAction,
        help="Should hops be considered?",
        default=False,
    )
    parser.add_argument(
        "--to-datetime",
        action=argparse.BooleanOptionalAction,
        help="Should numerical timestep be converted to datetime?",
        default=True,
    )
    parser.add_argument(
        "--add-duration",
        action=argparse.BooleanOptionalAction,
        help="Should a duration of 5mins be added?",
        default=True,
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    p = DATA_FOLDER / f"{args.filename}.csv"
    df = pd.read_csv(p, names=["datetime", "source", "destination"])
    if args.to_datetime:
        df = to_datetime(df)
    start_date, end_date = df["datetime"].min(), df["datetime"].max()
    if args.dataset_metadata:
        add_to_datasets(df, start_date=start_date, end_date=end_date)
    if args.groups:
        df = group_by_duration(df)
    if args.snapshot:
        G = nx.from_pandas_edgelist(
            df[df["datetime"] == start_date], target="destination"
        )
        show_graph(G, labels=True)
    if args.groups:
        df = df.drop(columns=["source", "destination"])
    if args.add_duration:
        df["duration"] = 5
    p.rename(Path(p.parent, f"{p.stem}_raw{p.suffix}"))
    df.to_csv(DATA_FOLDER / f"{args.filename}.csv", index=False)
