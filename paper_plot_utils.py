from pathlib import Path

import matplotlib


PAPER_RASTER_DPI = 400
PAPER_HATCHES = ("", "///", "\\\\\\", "xx", "..", "--", "++", "oo")


def configure_paper_plots():
    """Use IJCAI-style figure defaults for paper-ready PDF output."""
    matplotlib.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 10,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "patch.linewidth": 0.6,
            "savefig.bbox": "tight",
            "savefig.dpi": PAPER_RASTER_DPI,
            "savefig.pad_inches": 0.02,
        }
    )


def apply_bar_hatches(bars, hatches=PAPER_HATCHES):
    for index, bar in enumerate(bars):
        bar.set_hatch(hatches[index % len(hatches)])
        bar.set_edgecolor("#111111")
        bar.set_linewidth(0.6)


def save_paper_figure(fig, output_path, *, keep_png=True, dpi=PAPER_RASTER_DPI, **savefig_kwargs):
    """Save a paper-ready PDF plus, by default, the legacy PNG path."""
    output_path = Path(output_path)
    savefig_kwargs.setdefault("bbox_inches", "tight")
    savefig_kwargs.setdefault("pad_inches", 0.02)

    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", dpi=dpi, **savefig_kwargs)

    if keep_png and output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path, dpi=dpi, **savefig_kwargs)

    return pdf_path


configure_paper_plots()
