"""Journal figure style (Journal of Bionic Engineering instructions + editor's requests):
Times New Roman, text >= 7.5 pt at final size, lines >= 0.3 pt, sentence-case wording,
174 mm (full width) or 84 mm (column) figures, 600 dpi raster + vector PDF,
colour-blind-safe Okabe-Ito palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
FULL, COL = 174 * MM, 84 * MM
CLASS_NAMES = ["Complicated", "Non-complicated", "Normal"]
CLASS_COLORS = ["#D55E00", "#0072B2", "#009E73"]          # vermilion, blue, bluish green
FOLD_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.unicode_minus": True,
})


def save(fig, path_no_ext):
    fig.savefig(path_no_ext + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path_no_ext + ".tif", dpi=600, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
