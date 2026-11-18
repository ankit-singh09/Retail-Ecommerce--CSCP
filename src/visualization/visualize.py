"""Reusable EDA visualization helpers for cleaned Online Retail II data."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def _save_fig(fig: plt.Figure, save_path: Optional[str | Path]) -> None:
    """Save figure to disk when a target path is provided."""
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")


def plot_monthly_revenue_trend(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Plot monthly net revenue highlighting the partial final month (2011-12)."""
    monthly = (
        df.groupby(["YearMonth", "Is_Partial_Month"], as_index=False)
        .agg(
            Net_Revenue=("TotalAmount", "sum"),
            Unique_Invoices=("Invoice", "nunique"),
        )
        .sort_values("YearMonth")
        .reset_index(drop=True)
    )

    colors = [
        "#DD8452" if is_partial else "#4C72B0"
        for is_partial in monthly["Is_Partial_Month"]
    ]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(monthly["YearMonth"], monthly["Net_Revenue"] / 1e6, color=colors)
    ax.set_title("Monthly Net Revenue Trend (Dec 2009 - Dec 2011)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Net Revenue (£ Millions)")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("£%.2fM"))
    ax.tick_params(axis="x", rotation=45)

    # Annotate partial month
    partial_idx = monthly.index[monthly["Is_Partial_Month"]].tolist()
    if partial_idx:
        idx = partial_idx[-1]
        val = monthly.loc[idx, "Net_Revenue"] / 1e6
        ax.annotate(
            "Partial month\n(through Dec 9)",
            xy=(idx, val),
            xytext=(idx - 0.3, val + 0.38),
            ha="center",
            arrowprops={"arrowstyle": "->", "color": "#DD8452"},
            fontsize=9,
            color="#B05020",
            fontweight="bold",
        )

    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return monthly


def plot_weekday_hourly_sales(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> tuple[pd.Series, pd.Series]:
    """Plot gross revenue by day of week and order count by hour of day."""
    active = df[~df["Is_Cancelled"]]
    dow_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ]
    dow_rev = active.groupby("DayOfWeek")["TotalAmount"].sum().reindex(dow_order, fill_value=0)
    hour_orders = active.groupby("Hour")["Invoice"].nunique().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    sns.barplot(x=dow_rev.index, y=dow_rev.values / 1e6, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Gross Revenue by Day of Week", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Gross Revenue (£ Millions)")
    axes[0].yaxis.set_major_formatter(mtick.FormatStrFormatter("£%.1fM"))
    axes[0].tick_params(axis="x", rotation=30)

    sns.lineplot(x=hour_orders.index, y=hour_orders.values, marker="o", ax=axes[1], color="#55A868")
    axes[1].set_title("Unique Completed Invoices by Hour of Day", fontweight="bold")
    axes[1].set_xlabel("Hour of Day (24h)")
    axes[1].set_ylabel("Unique Invoices")
    axes[1].set_xticks(hour_orders.index)

    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return dow_rev, hour_orders


def plot_top_merchandise_products(
    df: pd.DataFrame,
    top_n: int = 10,
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Plot top merchandise SKUs by net revenue and net units sold."""
    merch = df[df["Item_Category"] == "Merchandise"]
    prod_summary = (
        merch.groupby(["StockCode", "Description"], as_index=False)
        .agg(Net_Revenue=("TotalAmount", "sum"), Net_Units=("Quantity", "sum"))
        .sort_values("Net_Revenue", ascending=False)
    )

    top_rev = prod_summary.head(top_n).copy()
    top_units = prod_summary.sort_values("Net_Units", ascending=False).head(top_n).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.barplot(x=top_rev["Net_Revenue"] / 1e3, y=top_rev["Description"], ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"Top {top_n} Merchandise SKUs by Net Revenue", fontweight="bold")
    axes[0].set_xlabel("Net Revenue (£ Thousands)")
    axes[0].set_ylabel("")

    sns.barplot(x=top_units["Net_Units"] / 1e3, y=top_units["Description"], ax=axes[1], color="#55A868")
    axes[1].set_title(f"Top {top_n} Merchandise SKUs by Net Units Sold", fontweight="bold")
    axes[1].set_xlabel("Net Units Sold (Thousands)")
    axes[1].set_ylabel("")

    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return top_rev


def plot_registered_vs_guest_cancellations(
    df: pd.DataFrame,
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Compare net revenue and cancellation rate between registered and guest cohorts."""
    cohort = df.groupby("Is_Registered", as_index=False).agg(
        Total_Lines=("Invoice", "count"),
        Unique_Invoices=("Invoice", "nunique"),
        Net_Revenue=("TotalAmount", "sum"),
        Cancelled_Lines=("Is_Cancelled", "sum"),
    )
    cohort["Cohort"] = cohort["Is_Registered"].map({True: "Registered", False: "Guest"})
    cohort["Cancel_Rate_Pct"] = (cohort["Cancelled_Lines"] / cohort["Total_Lines"]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sns.barplot(
        x="Cohort",
        y=cohort["Net_Revenue"] / 1e6,
        data=cohort,
        ax=axes[0],
        palette=["#DD8452", "#4C72B0"],
        hue="Cohort",
        legend=False,
    )
    axes[0].set_title("Net Revenue: Registered vs Guest", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Net Revenue (£ Millions)")
    for i, val in enumerate(cohort["Net_Revenue"] / 1e6):
        axes[0].text(i, val + 0.3, f"£{val:.2f}M", ha="center", fontweight="bold")
    axes[0].set_ylim(0, (cohort["Net_Revenue"].max() / 1e6) * 1.18)

    sns.barplot(
        x="Cohort",
        y="Cancel_Rate_Pct",
        data=cohort,
        ax=axes[1],
        palette=["#DD8452", "#4C72B0"],
        hue="Cohort",
        legend=False,
    )
    axes[1].set_title("Line-Item Cancellation Rate (%)", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Cancelled Lines (%)")
    for i, val in enumerate(cohort["Cancel_Rate_Pct"]):
        axes[1].text(i, val + 0.06, f"{val:.2f}%", ha="center", fontweight="bold")
    axes[1].set_ylim(0, cohort["Cancel_Rate_Pct"].max() * 1.25)

    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return cohort
