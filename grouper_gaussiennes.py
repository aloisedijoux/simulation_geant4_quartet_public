
import re
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


BASE_DIR = Path("/home/aloiselkb/g4bl_simu/11042026/")  # dossier contenant les sous-dossiers "output_..."
MATERIAU = "Ti"


FIG_NAME    = "fig_fit_gaussienne.png"
OUTPUT_FILE = BASE_DIR / f"all_gaussiennes_{MATERIAU.lower()}.png"

if not BASE_DIR.is_dir():
    sys.exit(f"  Dossier introuvable : {BASE_DIR}")

#  Collecte 
def parse_params(folder_name):
    name = folder_name.removeprefix("output_")
    m_P = re.search(r'P(\d+)_',          name)
    m_p = re.search(r'_p(\d+)_',         name)
    m_t = re.search(r'_t([\d.]+)mm',     name)
    m_r = re.search(r'Radius([\d.]+)mm', name)
    P = int(m_P.group(1))     if m_P else 0
    p = int(m_p.group(1))     if m_p else 0
    t = float(m_t.group(1))   if m_t else 0.0
    r = float(m_r.group(1))   if m_r else 0.0
    return P, p, t, r

entries = []
for folder in sorted(BASE_DIR.iterdir()):
    if not folder.is_dir() or not folder.name.startswith("output_"):
        continue
    if MATERIAU not in folder.name:
        continue
    fig_path = folder / FIG_NAME
    if fig_path.exists():
        P, p, t, r = parse_params(folder.name)
        entries.append((P, p, t, r, fig_path))

if not entries:
    sys.exit("  Aucune figure trouvée.")

#  Détection automatique des axes 
# Les 2 paramètres avec le plus de valeurs uniques → axes de la grille
# Le paramètre restant (souvent fixe) → info dans le titre global

params = {
    "P [mbar]":    sorted(set(e[0] for e in entries)),
    "p [MeV/c]":   sorted(set(e[1] for e in entries)),
    "t [mm]":      sorted(set(e[2] for e in entries)),
    "R [mm]":      sorted(set(e[3] for e in entries)),
}

# Tri par nombre de valeurs uniques décroissant
sorted_params = sorted(params.items(), key=lambda x: len(x[1]), reverse=True)

row_label, row_values = sorted_params[0]   # plus de variation → lignes
col_label, col_values = sorted_params[1]   # 2e variation     → colonnes
fix_params = sorted_params[2:]             # paramètres fixes → titre

# Accès rapide aux valeurs brutes par nom
def get_val(entry, label):
    if label == "P [mbar]":  return entry[0]
    if label == "p [MeV/c]": return entry[1]
    if label == "t [mm]":    return entry[2]
    return entry[3]  # R [mm]

img_map = {(get_val(e, row_label), get_val(e, col_label)): e[4] for e in entries}
n_rows, n_cols = len(row_values), len(col_values)

print(f"Figures trouvées : {len(entries)}")
print(f"Lignes   ({row_label}) : {row_values}  →  {n_rows}")
print(f"Colonnes ({col_label}) : {col_values}  →  {n_cols}")
for fix_label, fix_values in fix_params:
    print(f"Fixe     ({fix_label}) : {fix_values}")
print(f"Grille   : {n_rows} × {n_cols}")

#  Figure 
CELL_W, CELL_H, TOP_PAD = 5.2, 3.2, 0.6
fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(CELL_W * n_cols, CELL_H * n_rows + TOP_PAD),
    squeeze=False, dpi=300,
)
fig.patch.set_facecolor("white")

for row_idx, rv in enumerate(row_values):
    for col_idx, cv in enumerate(col_values):
        ax  = axes[row_idx][col_idx]
        key = (rv, cv)
        if key in img_map:
            ax.imshow(mpimg.imread(str(img_map[key])), interpolation="lanczos")
            ax.set_title(f"{row_label} = {rv}  |  {col_label} = {cv}",
                         fontsize=9, pad=5, fontweight="semibold", color="#1a1a2e")
        else:
            ax.set_facecolor("#f5f5f5")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    fontsize=13, color="#bbbbbb", transform=ax.transAxes)
            ax.set_title(f"{row_label} = {rv}  |  {col_label} = {cv}",
                         fontsize=9, pad=5, color="#bbbbbb")
        ax.axis("off")

fix_str = "  |  ".join(f"{lbl} = {', '.join(str(v) for v in vals)}" for lbl, vals in fix_params)
fig.suptitle(
    f"Gaussian fits — Material: {MATERIAU}  |  {fix_str}",
    fontsize=15, fontweight="bold", color="#1a1a2e",
)
plt.tight_layout(rect=[0.03, 0, 1, 1 - TOP_PAD / (CELL_H * n_rows + TOP_PAD)])
plt.subplots_adjust(hspace=0.0, wspace=0.02)

for row_idx, rv in enumerate(row_values):
    pos = axes[row_idx][0].get_position()
    fig.text(0.001, (pos.y0 + pos.y1) / 2, f"{row_label} = {rv}",
             va="center", ha="left", fontsize=8.5,
             rotation=90, color="#3a3a6e", fontweight="semibold")

fig.savefig(OUTPUT_FILE, dpi=600, bbox_inches="tight", facecolor="white")
print(f"\n  Figure sauvegardée : {OUTPUT_FILE}")
plt.close(fig)
