################################################################################
#                         MxL-GEN - Surrogate Data Exploration                 #
#                                                                              #
# Small utilities to inspect and summarise surrogate training datasets.       #
# Loads CSVs, cleans constant columns, scales data for PCA, and writes        #
# plots and CSV diagnostics under `surrogateCreation`.                        #
#                                                                              #
################################################################################
"""
Notes:
    - Small helper class for quick exploratory data analysis (EDA) on surrogate
      datasets (inputs and outputs).
    - Loads CSVs, removes constant columns, fits a StandardScaler, and stores
      scaled data for PCA and modeling.
    - Produces visual outputs and CSV diagnostics under the `surrogateCreation`
      folder for easy inspection.
"""

__all__ = ["data_exploration"]

import os
from contextlib import contextmanager
import warnings
from typing import Optional, Sequence, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Ensure the target folder for outputs exists
os.makedirs("surrogateCreation", exist_ok=True)


class data_exploration:
    """Quick exploratory data analysis helper for surrogate input/output datasets.

    The class loads input/output CSVs, removes constant columns, fits a
    StandardScaler, and provides plotting and diagnostic helpers. Outputs are
    written under ``surrogateCreation``.

    Args:
        input_data_path (str): Path to CSV of input parameters (no header).
        output_data_path (str): Path to CSV of output values (no header).
        label (str, optional): Prefix used for saved files. Defaults to "".
        scaled_vis (bool, optional): If True, visualisations use scaled data.
            Defaults to True.
    """

    def __init__(
        self,
        input_data_path: str,
        output_data_path: str,
        label: str = "",
        scaled_vis: bool = True,
    ) -> None:

        self.input_path = input_data_path
        self.output_path = output_data_path
        self.label = label or ""
        self.scaled_vis = bool(scaled_vis)
        self.log_file = os.path.join("surrogateCreation", f"{self.label}logFile.txt")

        # Load input and output CSVs; raise a clear error if missing
        try:
            self.input_df = pd.read_csv(self.input_path, header=None)
            self.output_data = pd.read_csv(self.output_path, header=None)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Input/output files missing: {e}. Ensure CSV paths are correct.")

        # Count NaNs before cleaning
        nan_count = int(self.input_df.isna().sum().sum())

        # Identify and remove constant columns (no effect on downstream numeric ops)
        constant_cols = self.input_df.nunique() == 1
        constant_indices = constant_cols[constant_cols].index.tolist()
        if constant_indices:
            self.input_df.drop(columns=constant_indices, inplace=True)

        # Overwrite the input CSV with the cleaned version so downstream steps see it
        self.input_df.to_csv(self.input_path, index=False, header=None)

        # Fit a StandardScaler once and store the scaled numpy array for PCA/ML tasks
        self.scaler = StandardScaler()
        self.input_data_scaled = self.scaler.fit_transform(self.input_df)

        # Ensure output folder exists (idempotent)
        os.makedirs("surrogateCreation", exist_ok=True)

        # Write a short summary log for traceability
        with open(self.log_file, "w") as f:
            f.write(f"Input data summary for '{self.label}'\n")
            f.write(f"{self.input_df.shape[1]} features after cleaning.\n")
            f.write(f"{self.input_df.shape[0]} total samples.\n")
            f.write(f"Total NaN values found: {nan_count}\n")
            if constant_indices:
                f.write(f"Constant columns removed at indices: {constant_indices}\n")
            else:
                f.write("No constant columns found.\n")
            f.write(f"Visualisations will use {'scaled' if self.scaled_vis else 'unscaled'} data.\n")

        # Print a concise startup summary to the console
        print("Surrogate data loaded correctly.")
        print(f"Total NaN values: {nan_count}")
        print("Removed constant columns:", constant_indices)
        print(f"Visualisations will use {'scaled' if self.scaled_vis else 'unscaled'} data.")

    # ---------------------------------------------------------------------

    @contextmanager
    def _suppress_seaborn_warnings(self) -> Iterator[None]:
        """Temporarily suppress seaborn/pandas plotting warnings.

        Returns:
            Iterator[None]: Yields control while warnings are filtered.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r"seaborn.*")
            warnings.filterwarnings("ignore", category=FutureWarning, module=r"seaborn.*")
            warnings.filterwarnings("ignore", category=FutureWarning, module=r"pandas.*")
            yield

    # ---------------------------------------------------------------------

    def _df_for_visualisation(self) -> pd.DataFrame:
        """Get the DataFrame used for plotting, scaled or raw.

        Returns:
            pd.DataFrame: DataFrame ready for visualisation with +/-inf replaced.
        """
        if self.scaled_vis:
            return pd.DataFrame(self.input_data_scaled).replace([np.inf, -np.inf], np.nan)
        return self.input_df.replace([np.inf, -np.inf], np.nan)

    # ---------------------------------------------------------------------

    def correlation_matrix(self) -> None:
        """Plot and save a correlation matrix heatmap.

        The image is written to `surrogateCreation/<label>correlation_matrix.png`
        and the action is appended to the instance log.
        """
        df = self._df_for_visualisation()
        corr = df.corr()

        plt.figure(figsize=(12, 10))
        with self._suppress_seaborn_warnings():
            sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title("Correlation Matrix")
        plt.tight_layout()

        out_file = os.path.join("surrogateCreation", f"{self.label}correlation_matrix.png")
        plt.savefig(out_file)
        plt.close()

        with open(self.log_file, "a") as f:
            f.write(f"Correlation matrix saved to {out_file}\n")

    # ---------------------------------------------------------------------

    def explanatory_dimension(self, explained_proportion: float = 0.95) -> np.ndarray:
        """Run PCA on scaled data and return projection keeping enough components.

        Args:
            explained_proportion (float, optional): Fraction of variance to retain.
                Defaults to 0.95.

        Returns:
            np.ndarray: Data projected onto the selected principal components.
        """
        pca = PCA()
        pca.fit(self.input_data_scaled)
        cum = np.cumsum(pca.explained_variance_ratio_)
        n = int(np.searchsorted(cum, explained_proportion) + 1)

        out_plot = os.path.join("surrogateCreation", f"{self.label}explained_variance.png")

        plt.figure(figsize=(10, 6))
        plt.plot(cum, marker="o")
        plt.axvline(n, color="r", linestyle="--")
        plt.title("Cumulative Explained Variance")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Variance")
        plt.tight_layout()
        plt.savefig(out_plot)
        plt.close()

        with open(self.log_file, "a") as f:
            f.write(f"PCA: Retaining {n} components for {explained_proportion:.2f} variance.\n")
            f.write(f"Saved explained variance plot to {out_plot}\n")

        return PCA(n_components=n).fit_transform(self.input_data_scaled)

    # ---------------------------------------------------------------------

    def feature_histograms(
        self,
        cols: Optional[Sequence[int]] = None,
        bins: int = 30,
        sample: Optional[int] = 5000,
    ) -> None:
        """Plot histograms with KDE for selected features.

        Args:
            cols (Optional[Sequence[int]], optional): Column indices to plot.
                If None, all features are plotted.
            bins (int, optional): Number of histogram bins. Defaults to 30.
            sample (Optional[int], optional): Max rows to sample for plotting.
                Defaults to 5000.

        Returns:
            None: Writes `<label>feature_histograms.png` to `surrogateCreation`.
        """
        df = self._df_for_visualisation()
        if sample is not None and df.shape[0] > sample:
            df = df.sample(sample, random_state=0)

        cols = cols or list(range(df.shape[1]))
        n = len(cols)
        ncols = 3
        nrows = int(np.ceil(n / ncols))

        plt.figure(figsize=(4 * ncols, 3 * nrows))
        for i, col in enumerate(cols):
            ax = plt.subplot(nrows, ncols, i + 1)
            with self._suppress_seaborn_warnings():
                sns.histplot(df[col], bins=bins, kde=True, stat="density")
            ax.set_title(f"Feature {col}")
        plt.tight_layout()

        out_file = os.path.join("surrogateCreation", f"{self.label}feature_histograms.png")
        plt.savefig(out_file)
        plt.close()

        with open(self.log_file, "a") as f:
            f.write(f"Feature histograms saved to {out_file}\n")

    # ---------------------------------------------------------------------

    def missing_value_report(self) -> pd.DataFrame:
        """Create and save a summary table of missing values.

        Returns:
            pd.DataFrame: Table with feature index, number missing and percent missing.

        Side effects:
            Writes `<label>missing_value_report.csv` to `surrogateCreation`.
        """
        df = self.input_df
        miss = df.isna().sum()
        pct = miss / len(df) * 100

        report = pd.DataFrame(
            {
                "feature": miss.index,
                "n_missing": miss.values,
                "pct_missing": pct.values,
            }
        )

        out_file = os.path.join("surrogateCreation", f"{self.label}missing_value_report.csv")
        report.to_csv(out_file, index=False)

        with open(self.log_file, "a") as f:
            f.write(f"Missing value report saved to {out_file}\n")

        return report

    # ---------------------------------------------------------------------

    def pairplot(self, sample: int = 500, hue: Optional[int] = None) -> None:
        """Generate a seaborn pairplot for a random subset of features.

        Args:
            sample (int, optional): Number of samples to draw. Defaults to 500.
            hue (Optional[int], optional): Index into `output_data` to use for hue.
                If provided, that output column is sampled for coloring.

        Raises:
            IndexError: If `hue` is out of range for `output_data`.
        """
        df = self._df_for_visualisation()

        nrows = df.shape[0]
        sample_n = min(sample, nrows)
        sample_idx = df.sample(sample_n, random_state=0).index
        df_sample = df.loc[sample_idx].reset_index(drop=True)

        if hue is not None:
            if hue >= self.output_data.shape[1]:
                raise IndexError("Selected hue column is out of range for output_data")
            hue_series = pd.Series(self.output_data.iloc[:, hue]).iloc[sample_idx].reset_index(drop=True)
            df_sample["hue"] = hue_series
            with self._suppress_seaborn_warnings():
                grid = sns.pairplot(df_sample, hue="hue", corner=True)
        else:
            with self._suppress_seaborn_warnings():
                grid = sns.pairplot(df_sample, corner=True)

        out_file = os.path.join("surrogateCreation", f"{self.label}pairplot.png")
        try:
            grid.savefig(out_file)
        except Exception:
            # fallback in case grid lacks savefig (older seaborn versions)
            plt.savefig(out_file)
        plt.close()

        with open(self.log_file, "a") as f:
            f.write(f"Wrote pairplot to {out_file}\n")

    # ---------------------------------------------------------------------

    def outlier_detection(self, method: str = "zscore", thresh: float = 3.0) -> np.ndarray:
        """Identify outlier rows using either z-score or IQR.

        Args:
            method (str, optional): 'zscore' or 'iqr'. Defaults to 'zscore'.
            thresh (float, optional): Threshold for determining outliers.
                Defaults to 3.0.

        Returns:
            np.ndarray: Boolean mask where True indicates an outlier row.

        Side effects:
            Saves `<label>outlier_indices.csv` listing detected row indices.
        """
        df = pd.DataFrame(self.input_data_scaled)

        if method == "zscore":
            z = np.abs((df - df.mean()) / df.std(ddof=0))
            mask = (z > thresh).any(axis=1).values
        elif method == "iqr":
            Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - thresh * IQR
            upper = Q3 + thresh * IQR
            mask = ((df < lower) | (df > upper)).any(axis=1).values
        else:
            raise ValueError("method must be 'zscore' or 'iqr'")

        idx = np.where(mask)[0]
        out_file = os.path.join("surrogateCreation", f"{self.label}outlier_indices.csv")
        pd.DataFrame({"outlier_index": idx}).to_csv(out_file, index=False)

        with open(self.log_file, "a") as f:
            f.write(f"Outliers detected ({len(idx)} rows). Saved to {out_file}\n")

        return mask

    # ---------------------------------------------------------------------

    def feature_importance_proxy(
        self,
        target_column: int = 0,
        n_estimators: int = 100,
        random_state: int = 0,
    ) -> pd.Series:
        """Quick proxy feature importance using RandomForestRegressor on scaled data.

        Args:
            target_column (int, optional): Output column index to use as the target.
                Defaults to 0.
            n_estimators (int, optional): Number of trees. Defaults to 100.
            random_state (int, optional): RNG seed for reproducibility. Defaults to 0.

        Returns:
            pd.Series: Sorted series of feature importances (index = feature column).

        Raises:
            IndexError: If `target_column` is out of range for `output_data`.

        Side effects:
            Writes `<label>feature_importances.csv` to `surrogateCreation`.
        """
        if target_column >= self.output_data.shape[1]:
            raise IndexError("target_column out of range.")

        X = pd.DataFrame(self.input_data_scaled)
        y = self.output_data.iloc[:, target_column].values

        # ensure same length for X and y
        n = min(len(X), len(y))
        X, y = X.iloc[:n, :], y[:n]

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X, y)

        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        out_file = os.path.join("surrogateCreation", f"{self.label}feature_importances.csv")
        imp.to_csv(out_file, header=["importance"])

        with open(self.log_file, "a") as f:
            f.write(f"Feature importance proxy saved to {out_file}\n")

        return imp

    # ---------------------------------------------------------------------

    def explain_variance_table(self) -> pd.DataFrame:
        """Return a detailed PCA variance breakdown table computed on scaled data.

        Returns:
            pd.DataFrame: Table with component index, explained_variance and cumulative_variance.

        Side effects:
            Writes `<label>explained_variance_table.csv` to `surrogateCreation`.
        """
        pca = PCA()
        pca.fit(self.input_data_scaled)
        ev = pca.explained_variance_ratio_
        cum = np.cumsum(ev)

        df = pd.DataFrame(
            {
                "component": np.arange(1, len(ev) + 1),
                "explained_variance": ev,
                "cumulative_variance": cum,
            }
        )

        out_file = os.path.join("surrogateCreation", f"{self.label}explained_variance_table.csv")
        df.to_csv(out_file, index=False)

        with open(self.log_file, "a") as f:
            f.write(f"PCA variance table saved to {out_file}\n")

        return df
