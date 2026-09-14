"""
Weight of Evidence (WOE) and Information Value (IV) Engine.
Basel II / Credit Risk Industry Standard Feature Engineering.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin

class WOEIVTransformer(BaseEstimator, TransformerMixin):
    """
    Computes Weight of Evidence (WOE) and Information Value (IV) for continuous
    and categorical variables, binning them monotonically where possible.

    Formulas:
    WOE = ln( (% Good / % Bad) )
    IV  = sum( (% Good - % Bad) * WOE )

    Note:
    In credit risk, Good = 0 (paid/current), Bad = 1 (default/charge-off).
    A positive WOE means higher proportion of Goods (lower risk).
    A negative WOE means higher proportion of Bads (higher risk).
    """

    def __init__(self, max_bins: int = 5, min_sample_pct: float = 0.05):
        self.max_bins = max_bins
        self.min_sample_pct = min_sample_pct
        self.woe_dict: Dict[str, Dict[str, Any]] = {}
        self.iv_summary: Dict[str, float] = {}

    def _bin_continuous(self, series: pd.Series, y: pd.Series) -> pd.Series:
        """Bins continuous variable using quantiles, handling NaNs as a dedicated bin."""
        clean_s = series.dropna()
        if len(clean_s.unique()) <= self.max_bins:
            binned = series.astype(str)
            return binned.fillna("Missing")

        try:
            # Generate quantile bins
            _, bin_edges = pd.qcut(clean_s, q=self.max_bins, retbins=True, duplicates="drop")
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            binned = pd.cut(series, bins=bin_edges)
            binned_str = binned.astype(str)
            return binned_str.fillna("Missing")
        except Exception:
            # Fallback to cut with uniform spacing
            binned = pd.cut(series, bins=self.max_bins)
            return binned.astype(str).fillna("Missing")

    def _bin_categorical(self, series: pd.Series) -> pd.Series:
        """Groups rare categories if needed, handles missing values."""
        s = series.fillna("Missing").astype(str)
        counts = s.value_counts(normalize=True)
        rare_cats = counts[counts < self.min_sample_pct].index
        if len(rare_cats) > 1:
            s = s.replace({cat: "Other" for cat in rare_cats})
        return s

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits WOE rules and calculates IV for each feature in X."""
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).values
        
        total_good = np.sum(y == 0)
        total_bad = np.sum(y == 1)

        if total_good == 0 or total_bad == 0:
            raise ValueError("Target must contain both 0 (Good) and 1 (Bad) classes.")

        self.woe_dict = {}
        self.iv_summary = {}

        for col in X.columns:
            series = X[col]
            is_numeric = pd.api.types.is_numeric_dtype(series)

            if is_numeric and series.nunique() > 10:
                binned = self._bin_continuous(series, y)
                var_type = "numeric"
            else:
                binned = self._bin_categorical(series)
                var_type = "categorical"

            # Create cross-tabulation of Bin vs Target
            df_bin = pd.DataFrame({"bin": binned, "target": y})
            grouped = df_bin.groupby("bin")["target"].agg(
                count="count",
                bad="sum"
            )
            grouped["good"] = grouped["count"] - grouped["bad"]

            # Laplace smoothing / adjustment for zero bads or goods
            grouped["good_adj"] = np.where(grouped["good"] == 0, 0.5, grouped["good"])
            grouped["bad_adj"] = np.where(grouped["bad"] == 0, 0.5, grouped["bad"])

            grouped["dist_good"] = grouped["good_adj"] / total_good
            grouped["dist_bad"] = grouped["bad_adj"] / total_bad

            grouped["woe"] = np.log(grouped["dist_good"] / grouped["dist_bad"])
            grouped["iv"] = (grouped["dist_good"] - grouped["dist_bad"]) * grouped["woe"]

            total_iv = grouped["iv"].sum()
            self.iv_summary[col] = float(total_iv)

            # Store mapping
            self.woe_dict[col] = {
                "var_type": var_type,
                "bin_map": grouped[["woe", "count", "good", "bad", "iv"]].to_dict(orient="index"),
                "total_iv": float(total_iv),
                "woe_map": grouped["woe"].to_dict()
            }

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input features X into Weight of Evidence (WOE) values."""
        X = pd.DataFrame(X).copy()
        X_woe = pd.DataFrame(index=X.index)

        for col in X.columns:
            if col not in self.woe_dict:
                continue

            info = self.woe_dict[col]
            var_type = info["var_type"]
            woe_map = info["woe_map"]

            series = X[col]
            if var_type == "numeric":
                # Re-bin using the bin keys
                # We match intervals
                binned = self._map_to_bins(series, list(woe_map.keys()))
            else:
                s = series.fillna("Missing").astype(str)
                binned = s.apply(lambda v: v if v in woe_map else ("Other" if "Other" in woe_map else "Missing"))

            # Replace bin labels with WOE values, filling unseen with median WOE (0.0 neutral)
            woe_series = binned.map(woe_map).fillna(0.0)
            X_woe[f"{col}_woe"] = woe_series

        return X_woe

    def _map_to_bins(self, series: pd.Series, bin_labels: List[str]) -> pd.Series:
        """Maps numeric values back into interval strings matching the fitted bins."""
        # Check for Missing bin
        res = pd.Series("Missing", index=series.index)
        not_null_mask = series.notna()
        vals = series[not_null_mask]

        interval_bins = []
        for bl in bin_labels:
            if bl in ["Missing", "Other"]:
                continue
            # format example: '(-inf, 12000.0]' or '(12000.0, 24000.0]'
            cleaned = bl.strip("[]() ")
            parts = cleaned.split(",")
            if len(parts) == 2:
                try:
                    left = float(parts[0].strip())
                    right = float(parts[1].strip())
                    interval_bins.append((left, right, bl))
                except ValueError:
                    pass

        # Sort intervals by left edge
        interval_bins.sort(key=lambda x: x[0])

        for left, right, label in interval_bins:
            mask = (vals > left) & (vals <= right)
            res.loc[vals[mask].index] = label

        return res

    def get_iv_summary_df(self) -> pd.DataFrame:
        """Returns Information Value table with credit industry predictive strength tiers."""
        rows = []
        for feature, iv in self.iv_summary.items():
            if iv < 0.02:
                strength = "Unpredictive (< 0.02)"
            elif iv < 0.10:
                strength = "Weak (0.02 - 0.10)"
            elif iv < 0.30:
                strength = "Medium (0.10 - 0.30)"
            elif iv < 0.50:
                strength = "Strong (0.30 - 0.50)"
            else:
                strength = "Suspiciously High (> 0.50)"
            rows.append({"feature": feature, "information_value": round(iv, 4), "predictive_power": strength})

        df_iv = pd.DataFrame(rows).sort_values(by="information_value", ascending=False)
        return df_iv
