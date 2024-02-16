import pandas as pd
import warnings

warnings.filterwarnings("ignore")

latex = 0
pand = 1
train_time = 0
test_time = 0

embeddings_order = [
    "bundle",
    "sequence",
    "ngram",
    "hashmap",
    "density",
    "flocet",
    "generic",
    "random",
    "sinusoid",
    "fractional",
]

df = pd.read_csv("survey_results/encodings_uci")

for i in df["encoding"].unique():
    variance_accuracy_by_dimension_and_method = (
        df[df["encoding"] == i].groupby(["name", "method"])["accuracy"].std()
    )

pivot_df = df.pivot_table(
    index="name", columns="encoding", values="accuracy", aggfunc="mean"
)
pivot_df = pivot_df.T
pivot_df["MEAN"] = pivot_df.mean(axis=1)
pivot_df = pivot_df.T.round(3)
pivot_df = pivot_df.reindex(columns=embeddings_order)
latex_table = pivot_df.to_latex()

print(latex_table)
