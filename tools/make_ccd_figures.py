#!/usr/bin/env python
# coding: utf-8
"""Generate academic-style figures for shared982 CCD summary.

Reads:
- cache/model_eval_results/shared982_ccd/ccd_summary.csv
Optionally joins:
- cache/model_eval_results/shared982_twoafc/twoafc_summary.csv
- cache/model_eval_results/shared982_rsa/rsa_summary.csv

Writes:
- cache/model_eval_results/shared982_ccd/figures/FigXX_*.png
- cache/model_eval_results/shared982_ccd/figures/figures_manifest.json

All figures are saved into a single folder and are numbered.
"""

from __future__ import annotations

import json
import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


PROJ = Path(__file__).resolve().parents[1]
CCD_DIR = Path(os.environ.get("CCD_DIR", str(PROJ / "cache" / "model_eval_results" / "shared982_ccd"))).resolve()
FIG_DIR = CCD_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CCD_CSV = CCD_DIR / "ccd_summary.csv"
TWOAFC_CSV = Path(
    os.environ.get(
        "TWOAFC_CSV",
        str(PROJ / "cache" / "model_eval_results" / "shared982_twoafc" / "twoafc_summary.csv"),
    )
).resolve()
RSA_CSV = Path(
    os.environ.get(
        "RSA_CSV",
        str(PROJ / "cache" / "model_eval_results" / "shared982_rsa" / "rsa_summary.csv"),
    )
).resolve()


def _setup_style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    try:
        import scienceplots  # noqa: F401

        import matplotlib.pyplot as plt

        plt.style.use(["science", "no-latex", "grid"])
    except Exception:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")

    # Enforce minimum font size > 7pt
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
    })


@dataclass
class FigItem:
    fig_id: int
    file: str
    title: str
    what: str
    comment: str
    usage: str


def _safe_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in s)


def _load_ccd() -> pd.DataFrame:
    if not CCD_CSV.is_file():
        raise RuntimeError(f"Missing {CCD_CSV}")
    df = pd.read_csv(CCD_CSV)
    # Focus on pooled_mean by default to avoid duplicated rows
    if "eval_repr" in df.columns:
        df = df[df["eval_repr"].isin(["pooled_mean", "tokens_mean"])].copy()
    return df


def _load_twoafc() -> Optional[pd.DataFrame]:
    if not TWOAFC_CSV.is_file():
        return None
    df = pd.read_csv(TWOAFC_CSV)
    # normalize eval_repr naming to match CCD
    df = df.rename(columns={"twoafc_fwd_mean": "twoafc"}).copy()
    df["eval_repr"] = df["eval_repr"].map({"pooled": "pooled_mean", "tokens_flatten": "tokens_mean"}).fillna(df["eval_repr"])
    return df[["group", "tag", "subj", "eval_repr", "twoafc"]]


def _load_rsa() -> Optional[pd.DataFrame]:
    if not RSA_CSV.is_file():
        return None
    df = pd.read_csv(RSA_CSV)
    return df[["group", "tag", "subj", "eval_repr", "rsa_pearson"]]


def fig01_per_subj_bars(df: pd.DataFrame) -> FigItem:
    import matplotlib.pyplot as plt
    import seaborn as sns

    d = df[df["eval_repr"] == "pooled_mean"].copy()
    d["label"] = d["group"].astype(str) + "/" + d["tag"].astype(str)

    subjs = sorted(d["subj"].dropna().astype(str).unique().tolist())
    n = len(subjs)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 6.0), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, s in zip(axes, subjs):
        ds = d[d["subj"].astype(str) == s].sort_values("ccd_acc1", ascending=True)
        sns.barplot(
            data=ds,
            y="label",
            x="ccd_acc1",
            ax=ax,
            color=sns.color_palette("deep")[0],
        )
        ax.set_title(f"subj{s}")
        ax.set_xlabel("CCD@1 (higher is better)")
        ax.set_ylabel("" if ax != axes[0] else "model")
        ax.set_xlim(0, 1)

    fig.suptitle("CCD@1 on shared982 (pooled_mean)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = FIG_DIR / "Fig01_ccd_acc1_per_subj_bar.png"
    fig.savefig(out)
    plt.close(fig)

    return FigItem(
        fig_id=1,
        file=str(out.relative_to(CCD_DIR)),
        title="CCD@1 per subject (bar)",
        what="每个被试一列横向条形图，展示各模型在 shared982 上的 CCD@1（pooled_mean）。",
        comment="用于快速看出同一被试下不同模型的 caption 判别能力差异；条形越长越好。",
        usage="优先在每个被试内挑选 CCD@1 靠前的模型作为 caption-alignment 的候选；若出现极低值，通常意味着特征空间或投影不匹配。",
    )


def fig02_acc1_vs_margin(df: pd.DataFrame) -> FigItem:
    import matplotlib.pyplot as plt
    import seaborn as sns

    d = df[df["eval_repr"] == "pooled_mean"].copy()
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.6))
    sns.scatterplot(
        data=d,
        x="margin_mean",
        y="ccd_acc1",
        hue="subj",
        style="group",
        s=70,
        ax=ax,
    )
    ax.set_title("CCD@1 vs mean margin (pooled_mean)")
    ax.set_xlabel("margin_mean (pos - max(neg))")
    ax.set_ylabel("CCD@1")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    out = FIG_DIR / "Fig02_ccd_acc1_vs_margin.png"
    fig.savefig(out)
    plt.close(fig)

    return FigItem(
        fig_id=2,
        file=str(out.relative_to(CCD_DIR)),
        title="CCD@1 vs margin_mean (scatter)",
        what="散点图：横轴为平均 margin（pos - hardest neg），纵轴为 CCD@1；按被试上色、按 group 变换 marker。",
        comment="一般 margin 越大，CCD@1 越高；如果二者脱钩，可能存在少量强负例主导或相似度分布异常。",
        usage="用来诊断：是整体排名提升（CCD@1↑）还是仅 margin 分布变化（可能受投影/归一化影响）。",
    )


def fig03_ccd_vs_twoafc(df: pd.DataFrame, twoafc: Optional[pd.DataFrame]) -> Optional[FigItem]:
    if twoafc is None:
        return None

    import matplotlib.pyplot as plt
    import seaborn as sns

    d = df[df["eval_repr"] == "pooled_mean"].copy()
    j = d.merge(twoafc[twoafc["eval_repr"] == "pooled_mean"], on=["group", "tag", "subj", "eval_repr"], how="inner")
    if j.empty:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.6))
    sns.scatterplot(data=j, x="twoafc", y="ccd_acc1", hue="subj", style="group", s=70, ax=ax)
    ax.set_title("CCD@1 vs 2AFC (pooled_mean)")
    ax.set_xlabel("2AFC forward mean")
    ax.set_ylabel("CCD@1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    out = FIG_DIR / "Fig03_ccd_acc1_vs_twoafc.png"
    fig.savefig(out)
    plt.close(fig)

    return FigItem(
        fig_id=3,
        file=str(out.relative_to(CCD_DIR)),
        title="CCD@1 vs 2AFC (scatter)",
        what="将 CCD@1 与 shared982 2AFC（forward）做散点对比（pooled_mean）。",
        comment="二者衡量的对齐对象不同：2AFC 是 image-image 近邻判别，CCD 是 caption 判别；相关性高说明 caption 能力随整体表示质量同步提升。",
        usage="用来判断：模型提升是否只体现在图像检索/2AFC，还是也能转化为 caption 判别能力。",
    )


def fig04_ccd_vs_rsa(df: pd.DataFrame, rsa: Optional[pd.DataFrame]) -> Optional[FigItem]:
    if rsa is None:
        return None

    import matplotlib.pyplot as plt
    import seaborn as sns

    d = df[df["eval_repr"] == "pooled_mean"].copy()
    j = d.merge(rsa[rsa["eval_repr"].isin(["pooled", "pooled_mean"])], on=["group", "tag", "subj"], how="inner")
    if j.empty:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.6))
    sns.scatterplot(data=j, x="rsa_pearson", y="ccd_acc1", hue="subj", style="group", s=70, ax=ax)
    ax.set_title("CCD@1 vs RSA(pearson)")
    ax.set_xlabel("RSA (pearson)")
    ax.set_ylabel("CCD@1")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    out = FIG_DIR / "Fig04_ccd_acc1_vs_rsa.png"
    fig.savefig(out)
    plt.close(fig)

    return FigItem(
        fig_id=4,
        file=str(out.relative_to(CCD_DIR)),
        title="CCD@1 vs RSA(pearson) (scatter)",
        what="将 CCD@1 与 shared982 RSA(pearson) 做散点对比（按被试上色）。",
        comment="若相关性较高，说明 caption 判别能力与表示几何一致性同向提升；若相关性弱，可能表明 caption 评测更依赖跨模态投影与文本编码细节。",
        usage="用来定位改进方向：如果 RSA 高但 CCD 低，优先检查文本侧（caption、tokenization、负样本）与 1664→1280 投影策略。",
    )


def fig05_group_box(df: pd.DataFrame) -> FigItem:
    import matplotlib.pyplot as plt
    import seaborn as sns

    d = df[df["eval_repr"] == "pooled_mean"].copy()
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.6))
    sns.boxplot(data=d, x="subj", y="ccd_acc1", hue="group", ax=ax)
    ax.set_title("CCD@1 distribution by group (pooled_mean)")
    ax.set_xlabel("subject")
    ax.set_ylabel("CCD@1")
    ax.set_ylim(0, 1)
    ax.legend(title="group", loc="best")
    fig.tight_layout()

    out = FIG_DIR / "Fig05_ccd_acc1_group_box.png"
    fig.savefig(out)
    plt.close(fig)

    return FigItem(
        fig_id=5,
        file=str(out.relative_to(CCD_DIR)),
        title="CCD@1 distribution by group (box)",
        what="按 subj 分组展示不同 group 的 CCD@1 分布（pooled_mean）。",
        comment="用于查看不同训练来源/导出来源的整体水平与稳定性；箱体更高、更紧通常更好。",
        usage="如果某个 group 在多个 subj 上整体偏低，建议优先排查其导出 embedding 是否处于同一 CLIP 空间/预处理一致。",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with_ci", action="store_true", help="Include CI where available (mainly affects descriptions)")
    ap.add_argument("--by_type", action="store_true", help="Include by-type figure if generated")
    ap.add_argument("--audit", action="store_true", help="Include audit figures if generated")
    ap.add_argument("--ablation", action="store_true", help="Include ablation figures if generated")
    args = ap.parse_args()

    _setup_style()
    df = _load_ccd()
    twoafc = _load_twoafc()
    rsa = _load_rsa()

    figs: List[FigItem] = []
    figs.append(fig01_per_subj_bars(df))
    figs.append(fig02_acc1_vs_margin(df))

    f3 = fig03_ccd_vs_twoafc(df, twoafc)
    if f3 is not None:
        figs.append(f3)

    f4 = fig04_ccd_vs_rsa(df, rsa)
    if f4 is not None:
        figs.append(f4)

    figs.append(fig05_group_box(df))

    # Optional: include external figures (generated by other scripts)
    next_id = max([f.fig_id for f in figs], default=0) + 1

    if args.by_type:
        p = FIG_DIR / "Fig06_ccd_by_type.png"
        if p.is_file():
            figs.append(
                FigItem(
                    fig_id=next_id,
                    file=str(p.relative_to(CCD_DIR)),
                    title="CCD-Hard by type (bar, with CI)",
                    what="按 hardneg 的 type（object/attribute/relation）分解 CCD@1；默认选择每个 subject 的最佳 pooled 模型，并用 bootstrap CI 画误差棒。",
                    comment="用于展示模型机制差异：object 往往更容易，attribute/relation 更考验细粒度与组合能力；CI 可用于说明差异是否稳健。",
                    usage="主文可用作机制图；补充材料可配合 ccd_by_type.csv 展示每个模型的分解结果。",
                )
            )
            next_id += 1

    if args.ablation:
        for fname, title, what, comment, usage in [
            (
                "Fig07_ccd_ablation_k.png",
                "CCD-Hard K ablation (line, with CI)",
                "hard negatives per image 的数量 K（例如 2/4）消融：展示 CCD@1 随 K 的变化（每个 subject 取最佳 pooled 模型）。",
                "K 增大通常会降低 CCD@1（任务更难）但更接近真实‘hard’判别；若模型在更大 K 下仍保持优势，说明对强负例更鲁棒。",
                "用于补充材料/消融章节，解释为什么主结果选择某个 K（或为何采用 hardest-neg 口径）。",
            ),
            (
                "Fig08_ccd_ablation_window.png",
                "CCD-Hard similarity-window ablation (bar, with CI)",
                "tight vs loose 相似度窗口消融：对候选 hardneg 以 sim_text 过滤后再评测（每个 subject 取最佳 pooled 模型）。",
                "如果 loose/tight 的趋势一致，说明结果不依赖某个特定阈值；若差异较大，需要在方法中明确窗口选择并解释。",
                "用于回答 reviewer 关于阈值选择/可操纵性的质疑。",
            ),
            (
                "Fig09_ccd_ablation_difficulty.png",
                "CCD-Hard difficulty ablation (bar, with CI)",
                "hardest（按 sim_text 取 top-K）vs random（从候选池随机采样 K）难度消融：展示 CCD@1 对‘负例选择策略’的敏感性（每个 subject 取最佳 pooled 模型）。",
                "用于验证主结论是否依赖某个特定的 hardneg 选择规则；注意 random 只使用文本侧 sim_text 与随机种子，禁止使用 brain 相似度。",
                "用于补充材料的机制/敏感性分析小节；配合 results/tables/ccd_ablation_difficulty.csv 给出数值。",
            ),
        ]:
            p = FIG_DIR / fname
            if p.is_file():
                figs.append(FigItem(fig_id=next_id, file=str(p.relative_to(CCD_DIR)), title=title, what=what, comment=comment, usage=usage))
                next_id += 1

    if args.audit:
        audit_dir = CCD_DIR / "audit"
        if audit_dir.is_dir():
            for p in sorted(audit_dir.glob("Fig_audit_*.png")):
                figs.append(
                    FigItem(
                        fig_id=next_id,
                        file=str(p.relative_to(CCD_DIR)),
                        title=f"Hardneg audit: {p.name}",
                        what="hard negative captions 的质量审计图（长度/否定词/sim_text 分布/type 覆盖率等）。",
                        comment="用于证明 hardneg 不是通过‘否定词’等投机技巧构造，且长度/相似度分布受控。",
                        usage="放补充材料的 Negative Quality Audit 小节；配合 audit_tables.csv 描述过滤与阈值。",
                    )
                )
                next_id += 1

    manifest = {
        "root": str(CCD_DIR),
        "fig_dir": str(FIG_DIR.relative_to(CCD_DIR)),
        "figures": [f.__dict__ for f in figs],
    }

    (FIG_DIR / "figures_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    # Also write a human-friendly report
    report_lines: List[str] = []
    report_lines.append("# CCD-Hard figures report\n\n")
    report_lines.append("本文件汇总所有图表的：名称、内容、评价、使用建议，并给出图像预览链接。\n\n")
    for f in figs:
        report_lines.append(f"## Figure {f.fig_id}: {f.title}\n\n")
        report_lines.append(f"- 内容：{f.what}\n")
        report_lines.append(f"- 评价：{f.comment}\n")
        report_lines.append(f"- 建议：{f.usage}\n\n")
        report_lines.append(f"![]({f.file})\n\n")
    (FIG_DIR / "figures_report.md").write_text("".join(report_lines), encoding="utf-8")
    print(f"[DONE] wrote {FIG_DIR / 'figures_manifest.json'} with {len(figs)} figures")


if __name__ == "__main__":
    main()
