
import os
import re
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


#  PARAMÈTRES GLOBAUX 

RACINE  = "/home/aloiselkb/g4bl_simu/11042026/"   # dossier racine contenant les sous-dossiers de config
FILTRE  = None   # sous-chaîne dans les noms de dossiers, ou None pour tous

L_GAZ   = 200.0    # longueur de la cellule gaz en mm
N_SCAN  = 400      # nombre de plans de scan
DZ_SCAN = L_GAZ / N_SCAN

SMOOTH_SIGMA = 4   # écart-type du filtre de lissage gaussien (en bins)
COL_ID = 8         # colonne EventID dans les fichiers ScanZ

# Critères de rejet du fit (physiquement non représentatif)
SIGMA_MAX_FRACTION = 1 / 3   # σ > L_GAZ * cette fraction → fit invalide
MU_MAX_FRACTION    = 0.90    # μ > L_GAZ * cette fraction → fit invalide


#  MATPLOTLIB 

plt.rcParams.update({
    "font.family"         : "serif",
    "font.serif"          : ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size"           : 12,
    "axes.titlesize"      : 13,
    "axes.labelsize"      : 13,
    "axes.linewidth"      : 1.1,
    "xtick.direction"     : "in",
    "ytick.direction"     : "in",
    "xtick.major.size"    : 5,
    "ytick.major.size"    : 5,
    "xtick.minor.visible" : True,
    "ytick.minor.visible" : True,
    "xtick.minor.size"    : 2.5,
    "ytick.minor.size"    : 2.5,
    "figure.dpi"          : 150,
    "savefig.dpi"         : 200,
    "savefig.bbox"        : "tight",
    "savefig.facecolor"   : "white",
})


#  UTILITAIRES 

def scan_idx_vers_z(idx: int) -> float:
    return (idx + 0.5) * DZ_SCAN


def gaussienne(x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def label_config(nom_config: str) -> str:
    m = re.search(r"P(?P<P>[\d.]+)_p(?P<p0>[\d.]+).*_t(?P<t>[\d.]+)mm", nom_config)
    if m:
        t_val = m.group("t").rstrip("0").rstrip(".")
        return (
            rf"$P = {m.group('P')}$ mbar, "
            rf"$p_0 = {m.group('p0')}$ MeV/c, "
            rf"$t = {t_val}$ mm"
        )
    return nom_config


def creer_dossier_sortie(dossier: str) -> str:
    nom = os.path.basename(dossier.rstrip("/"))
    sortie = os.path.join(os.path.dirname(dossier), f"output_{nom}")
    os.makedirs(sortie, exist_ok=True)
    return sortie


def sauvegarder_figure(fig, dossier_sortie: str, nom_fichier: str):
    fig.savefig(os.path.join(dossier_sortie, nom_fichier))
    plt.close(fig)


#  COLLECTE DES DOSSIERS 

def contient_fichiers_scan(chemin: str) -> bool:
    return any(
        f.startswith("ScanZ_") and f.endswith(".txt")
        for f in os.listdir(chemin)
    )


def collecter_dossiers(racine: str, filtre) -> list[str]:
    sous_filtres = (filtre,) if isinstance(filtre, str) else (filtre or ())
    dossiers = []
    for nom in sorted(os.listdir(racine)):
        if nom.startswith("output_"):
            continue
        chemin = os.path.join(racine, nom)
        if not os.path.isdir(chemin):
            continue
        if sous_filtres and not all(f in nom for f in sous_filtres):
            continue
        if not contient_fichiers_scan(chemin):
            continue
        dossiers.append(chemin)
    return dossiers


#  RECONSTRUCTION DE LA DISTRIBUTION 

def construire_distribution_arrets(dossier: str) -> pd.Series:
    events_par_scan: dict[int, set] = {}

    for nom in os.listdir(dossier):
        m = re.fullmatch(r"ScanZ_(\d+)\.txt", nom)
        if not m:
            continue
        idx = int(m.group(1))
        ids: set = set()
        with open(os.path.join(dossier, nom)) as fh:
            for ligne in fh:
                if ligne.startswith("#"):
                    continue
                parts = ligne.split()
                if len(parts) > COL_ID:
                    try:
                        ids.add(int(parts[COL_ID]))
                    except ValueError:
                        pass
        events_par_scan[idx] = ids

    scans      = sorted(events_par_scan)
    all_events = sorted(set().union(*events_par_scan.values()))

    df = pd.DataFrame(0, index=scans, columns=all_events, dtype=int)
    for s in scans:
        df.loc[s, list(events_par_scan[s])] = 1

    dernier_scan = df.index.max()
    dernier_par_event = df.apply(
        lambda col: col[col == 1].index.max() if (col == 1).any() else None
    )

    # Events arrêtés définitivement (avant le dernier scan)
    stops = dernier_par_event[dernier_par_event < dernier_scan].value_counts().sort_index()
    stops.index = stops.index.map(scan_idx_vers_z)

    # Events traversants (dernier scan atteint)
    n_traversants = (dernier_par_event == dernier_scan).sum()

    return stops, int(n_traversants)


#  FIT GAUSSIEN + FIGURE 

def _lissage_sur_grille(z_nat, y_nat):
    z_grid = np.linspace(DZ_SCAN / 2, L_GAZ - DZ_SCAN / 2, N_SCAN)
    y_grid = np.zeros(N_SCAN)
    for zi, yi in zip(z_nat, y_nat):
        idx = int(zi / DZ_SCAN)
        if 0 <= idx < N_SCAN:
            y_grid[idx] = yi
    return z_grid, gaussian_filter1d(y_grid, sigma=SMOOTH_SIGMA)


def _fit_valide(mu: float, sigma: float) -> tuple[bool, str]:
    """
    Vérifie que les paramètres du fit sont physiquement cohérents.
    Retourne (valide, raison_du_rejet).
    """
    if sigma > L_GAZ * SIGMA_MAX_FRACTION:
        return False, f"σ={sigma:.1f} mm > L_GAZ/{int(1/SIGMA_MAX_FRACTION)}={L_GAZ/3:.1f} mm (distribution too wide)"
    if mu > L_GAZ * MU_MAX_FRACTION:
        return False, f"μ={mu:.1f} mm > {MU_MAX_FRACTION*100:.0f}% of L_GAZ={L_GAZ*MU_MAX_FRACTION:.1f} mm (peak out of cell)"
    return True, ""


def fit_et_figure(
    distribution : pd.Series,
    n_traversants: int,
    dossier_sortie: str,
    nom_config    : str,
    log,
):
    z_nat = distribution.index.to_numpy(dtype=float)
    y_nat = distribution.values.astype(float)

    if len(z_nat) < 4:
        log("  SKIP : pas assez de points pour ajuster.")
        return

    #  Ajustement gaussien 
    sigma0 = (z_nat.max() - z_nat.min()) / 8.0
    try:
        popt, pcov = curve_fit(
            gaussienne, z_nat, y_nat,
            p0=[y_nat.max(), z_nat[np.argmax(y_nat)], sigma0],
            bounds=([0, 0, 0.5], [np.inf, L_GAZ, L_GAZ]),
            maxfev=20_000,
        )
        perr   = np.sqrt(np.diag(pcov))
        fit_ok = True
    except RuntimeError as exc:
        log(f"  Ajustement numérique échoué : {exc}")
        fit_ok = False

    #  Validation physique 
    fit_physique = False
    raison_rejet = ""
    if fit_ok:
        A, mu, sigma    = popt
        dA, dmu, dsigma = perr
        fwhm, dfwhm     = 2.3548 * sigma, 2.3548 * dsigma
        fit_physique, raison_rejet = _fit_valide(mu, sigma)

        if fit_physique:
            log(f"  μ    = {mu:.3f} ± {dmu:.3f} mm")
            log(f"  σ    = {sigma:.3f} ± {dsigma:.3f} mm")
            log(f"  FWHM = {fwhm:.3f} ± {dfwhm:.3f} mm")
            log(f"  Events traversants (non arrêtés) : {n_traversants}")
        else:
            log(f"  FIT_INVALIDE : {raison_rejet}")
            log(f"  (μ_fit={mu:.1f} mm, σ_fit={sigma:.1f} mm — non représentatif)")
            log(f"  Events traversants (non arrêtés) : {n_traversants}")

    #  Lissage pour la visualisation 
    z_grid, y_smooth = _lissage_sur_grille(z_nat, y_nat)

    #  Figure 
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.87, bottom=0.13)

    ax.fill_between(z_grid, y_smooth, alpha=0.25, color="#4878CF", zorder=2)
    ax.plot(z_grid, y_smooth, color="#4878CF", lw=1.5, alpha=0.8, zorder=3)

    if fit_ok and fit_physique:
        # Cas nominal : fit valide
        z_fin = np.linspace(0, L_GAZ, 3_000)
        ax.plot(z_fin, gaussienne(z_fin, A, mu, sigma), color="#C0392B", lw=2.2, zorder=5)
        ax.axvline(mu, color="#2C7A2C", lw=1.2, ls="--", alpha=0.85, zorder=4)

        boite = (
            rf"$\mu \;= {mu:.2f} \pm {dmu:.2f}$ mm"        + "\n"
            rf"$\sigma = {sigma:.2f} \pm {dsigma:.2f}$ mm"  + "\n"
            rf"$\mathrm{{FWHM}} = {fwhm:.2f} \pm {dfwhm:.2f}$ mm" + "\n"
            rf"$N_{{\mathrm{{traversants}}}} = {n_traversants}$"
        )
        boite_couleur = "#FAFAFA"
        boite_bord    = "#BBBBBB"

    else:
        # Cas dégénéré : distribution plate, muons qui traversent
        n_arretes = int(distribution.sum())
        boite = ("No significant stops in the cell"

rf"$N_{{\mathrm{{stops}}}} = {n_arretes}$" + "\n"

rf"$N_{{\mathrm{{crossings}}}} = {n_traversants}$"

)
        if fit_ok:
            boite += f"\n(fit rejected : {raison_rejet[:40]}…)" if len(raison_rejet) > 40 else f"\n(fit rejected : {raison_rejet})"
        boite_couleur = "#FFF4E5"   # fond orangé pour signaler le cas dégénéré
        boite_bord    = "#E8A020"

    ax.text(
        0.977, 0.96, boite,
        transform=ax.transAxes,
        va="top", ha="right",
        fontsize=10,
        linespacing=1.8,
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor=boite_couleur,
            edgecolor=boite_bord,
            linewidth=0.9,
        ),
        zorder=6,
    )

    ax.set_xlim(0, L_GAZ)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$z$ (mm)", labelpad=6)
    ax.set_ylabel("Number of stopped events", labelpad=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.grid(True, linestyle=":", linewidth=0.55, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        "Distribution of final stops\n" + label_config(nom_config),
        pad=10, linespacing=1.5,
    )

    sauvegarder_figure(fig, dossier_sortie, "fig_fit_gaussienne.png")
    log("  Figure saved: fig_fit_gaussienne.png")


#  MAIN LOOP 

def analyser_dossier(dossier: str):
    nom_config     = os.path.basename(dossier)
    dossier_sortie = creer_dossier_sortie(dossier)
    log_path       = os.path.join(dossier_sortie, "log_fit_gaussienne.txt")

    with open(log_path, "w") as log_fh:
        def log(msg=""):
            print(msg)
            log_fh.write(str(msg) + "\n")

        log(f"{''*60}")
        log(f" Gaussian fit : {nom_config}")
        log(f" L_gaz={L_GAZ} mm | N_scan={N_SCAN} | dz={DZ_SCAN} mm")
        log(f"{''*60}")

        log("\n[1/2] Reconstruction of the stop distribution")
        distribution, n_traversants = construire_distribution_arrets(dossier)
        log(f"  {len(distribution)} non-zero bins — {distribution.sum()} stopped events")
        log(f"  {n_traversants} crossing events (not stopped)")

        log("\n[2/2] Gaussian fitting")
        fit_et_figure(distribution, n_traversants, dossier_sortie, nom_config, log)

        log(f"\n{''*60}")
        log(" Done.")

    n = len(os.listdir(dossier_sortie))
    print(f"  → {n} file(s) produced in {dossier_sortie}")


def main():
    dossiers = collecter_dossiers(RACINE, FILTRE)
    print(f"{len(dossiers)} folder(s) to process:")
    for d in dossiers:
        print(f"  {os.path.basename(d)}")

    erreurs = []
    for i, dossier in enumerate(dossiers, 1):
        print(f"\n[{i}/{len(dossiers)}] {os.path.basename(dossier)}")
        try:
            analyser_dossier(dossier)
        except Exception as exc:
            print(f"  ERROR : {exc}")
            traceback.print_exc()
            erreurs.append((os.path.basename(dossier), str(exc)))

    print(f"\n{''*60}")
    print(f"Result : {len(dossiers) - len(erreurs)}/{len(dossiers)} folder(s) OK")
    if erreurs:
        print("Failures :")
        for nom, err in erreurs:
            print(f"  {nom} → {err}")


if __name__ == "__main__":
    main()
