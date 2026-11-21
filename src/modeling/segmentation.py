"""Customer segmentation pipeline using StandardScaler and K-Means clustering."""

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = ["Recency_Log", "Frequency_Log", "Monetary_Log"]

SEGMENT_ACTION_MAP = {
    "Champions / VIP": (
        "Reward with exclusive early access, tiered loyalty perks, and referral incentives; avoid unnecessary discounting."
    ),
    "At-Risk High-Value": (
        "Trigger personalized win-back campaigns with time-bound vouchers and account check-ins before full churn."
    ),
    "Promising / Active": (
        "Nurture via product recommendations, bundle cross-sells, and free-shipping thresholds to raise order frequency."
    ),
    "Dormant / Lost": (
        "Limit paid acquisition spend; run low-cost automated seasonal re-engagement emails to test reactivation."
    ),
}


def evaluate_cluster_range(
    X_scaled: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate Inertia, Silhouette Score, and Davies-Bouldin Index for k_min..k_max."""
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = km.fit_predict(X_scaled)
        rows.append({
            "k": k,
            "Inertia": round(float(km.inertia_), 2),
            "Silhouette_Score": round(float(silhouette_score(X_scaled, labels)), 4),
            "Davies_Bouldin": round(float(davies_bouldin_score(X_scaled, labels)), 4),
        })
    return pd.DataFrame(rows)


def _label_four_clusters(
    df_with_clusters: pd.DataFrame,
) -> Dict[int, str]:
    """Deterministically assign business persona names to 4 K-Means clusters."""
    stats = df_with_clusters.groupby("Cluster_ID").agg(
        Recency_Mean=("Recency_Days", "mean"),
        Freq_Mean=("Frequency", "mean"),
        Monetary_Mean=("Monetary", "mean"),
    )

    label_map: Dict[int, str] = {}
    # Highest monetary & frequency cluster is Champions / VIP
    champ_id = int(stats["Monetary_Mean"].idxmax())
    label_map[champ_id] = "Champions / VIP"

    remaining = stats.drop(index=champ_id)
    # Highest recency (longest inactive) & lowest frequency among remaining is Dormant / Lost
    dormant_id = int(remaining["Recency_Mean"].idxmax())
    label_map[dormant_id] = "Dormant / Lost"

    remaining = remaining.drop(index=dormant_id)
    # Between the two middle clusters, higher Recency (inactive) & higher Monetary is At-Risk High-Value
    at_risk_id = int(remaining["Recency_Mean"].idxmax())
    promising_id = int(remaining.drop(index=at_risk_id).index[0])
    label_map[at_risk_id] = "At-Risk High-Value"
    label_map[promising_id] = "Promising / Active"
    return label_map


def fit_kmeans_segmentation(
    rfm_df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Scale log RFM features, fit K-Means, and return segmented data + artifacts."""
    df_seg = rfm_df.copy()
    X = df_seg[CLUSTER_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    eval_df = evaluate_cluster_range(X_scaled, k_min=2, k_max=8, random_state=random_state)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    df_seg["Cluster_ID"] = kmeans.fit_predict(X_scaled)

    # Store standardized features for diagnostic plotting (e.g., snake plots)
    scaled_df = pd.DataFrame(
        X_scaled,
        columns=[f"{col}_Scaled" for col in CLUSTER_FEATURES],
        index=df_seg.index,
    )
    df_seg = pd.concat([df_seg, scaled_df], axis=1)

    if n_clusters == 4:
        name_map = _label_four_clusters(df_seg)
    else:
        ranked = (
            df_seg.groupby("Cluster_ID")["Monetary"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        name_map = {cid: f"Segment_{i + 1}" for i, cid in enumerate(ranked)}

    df_seg["Customer_Segment"] = df_seg["Cluster_ID"].map(name_map)
    df_seg["Recommended_Action"] = df_seg["Customer_Segment"].map(
        lambda s: SEGMENT_ACTION_MAP.get(s, "Monitor engagement and retention.")
    )

    artifact = {
        "scaler": scaler,
        "kmeans": kmeans,
        "feature_names": CLUSTER_FEATURES,
        "cluster_name_map": name_map,
        "silhouette_score": float(
            eval_df.loc[eval_df["k"] == n_clusters, "Silhouette_Score"].iloc[0]
        ),
    }
    return df_seg, eval_df, artifact


def main() -> None:
    """Run RFM-to-segmented customer clustering pipeline."""
    project_dir = Path(__file__).resolve().parents[2]
    rfm_in = project_dir / "data" / "processed" / "customer_rfm_features.csv"
    seg_out = project_dir / "data" / "processed" / "customer_segments.csv"
    model_out = project_dir / "models" / "kmeans_segmentation.pkl"

    rfm_df = pd.read_csv(rfm_in)
    df_seg, eval_df, artifact = fit_kmeans_segmentation(rfm_df, n_clusters=4)

    seg_out.parent.mkdir(parents=True, exist_ok=True)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    df_seg.to_csv(seg_out, index=False)
    joblib.dump(artifact, model_out)

    print("Cluster Evaluation (k=2..8):")
    print(eval_df.to_string(index=False))
    print(f"\nSaved segmented customers: {seg_out} ({df_seg.shape[0]:,} x {df_seg.shape[1]})")
    print(f"Saved K-Means artifact:    {model_out}")


if __name__ == "__main__":
    main()
