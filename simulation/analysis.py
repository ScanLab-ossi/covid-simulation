import pandas as pd
import numpy as np

from simulation.output import Output
from simulation.building_blocks import OutputBasicBlock


class Analysis(OutputBasicBlock):
    def sick(self, output: Output, what: str = "max_amount") -> pd.DataFrame:
        fname = "sick"
        summed = [
            output.sum_output(df).pivot(index="day", columns="color")["amount"]
            for df in output.batch
        ]
        bpr = [df["b"] + df["p"] + df["r"] for df in summed]
        if what == "max_amount":
            return pd.DataFrame([df.max() for df in bpr], columns=[fname])
        elif what == "max_day":
            return pd.DataFrame([df.idxmax() for df in bpr], columns=[fname])
        elif what == "total":
            return pd.DataFrame([len(df) for df in output.batch], columns=[fname])

    def infected(self, output: Output, what: str = "amount") -> pd.DataFrame:
        fname = "infected"
        grouped = [
            df.groupby("infection_date")["final_state"].count().reset_index(drop=True)
            for df in output.batch
        ]
        if what == "max_amount":
            return pd.DataFrame([df.max() for df in grouped], columns=[fname])
        elif what == "max_day":
            return pd.DataFrame([df.idxmax() for df in grouped], columns=[fname])

    def r_0(self, output: Output, what: str = "total") -> pd.DataFrame:
        fname = "r_0"
        if what == "total":
            sick = self.sick(output, "total")
            infectors = np.array(
                [len(set().union(*df["infector"].dropna())) for df in output.batch]
            )
            return pd.DataFrame(sick["sick"].values / infectors, columns=[fname])
        elif what == "average":
            return (
                pd.concat([self.basic_r_0(df) for df in output.batch])
                .groupby("infection_date")[["r_0", "r_thresh"]]
                .mean()
                .reset_index()
            )

    def basic_r_0(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.groupby("infection_date").agg(
            {
                "infector": lambda x: len(set().union(*x.dropna())),
                "final_state": "count",
            }
        )
        df = df.assign(r_0=df["final_state"] / df["infector"])
        df.index = df.index.astype("datetime64[ns]", copy=False)
        idx = pd.date_range(self.dataset.start_date, self.dataset.end_date)
        df = (
            df.reindex(idx, fill_value=0)
            .rename_axis("infection_date", axis="index")
            .reset_index()
        )
        df["r_thresh"] = 1
        return df

