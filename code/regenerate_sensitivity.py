import pandas as pd
from sensitivity_analysis import plot_sensitivity_heatmap

summary = pd.read_csv("results/sensitivity_summary.csv")
plot_sensitivity_heatmap(summary, save_path="figures/sensitivity_grid.png")