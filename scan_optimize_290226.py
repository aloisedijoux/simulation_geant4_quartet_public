import subprocess
import csv
import statistics as stats
from pathlib import Path
from itertools import product
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# PATHS (à adapter)

from pathlib import Path

from modularized_code import metrics
from modularized_code.g4bl_muon.scan_optimize_160326 import fmt

G4BL      = ".../G4beamline-3.08/bin/g4bl"
G4BL_FILE = "/home/aloiselkb/g4bl_simu/setup_config_linux.g4bl"

BASE_OUT = Path("/home/aloiselkb/g4bl_simu/forMike/")  # dossier de sortie pour les runs (un sous-dossier par config)
BASE_OUT.mkdir(parents=True, exist_ok=True)
NSCAN = 200
SCAN_FILES = [f"ScanZ_{i}.txt" for i in range(NSCAN)]

# Fichiers diagnostics ajoutés dans le .g4bl
DIAG_FILES = {
    "veto_in":   "Veto_in.txt",
    "veto_out":  "Veto_out.txt",
    "scint_in":  "Scint_in.txt",
    "scint_out": "Scint_out.txt",
    "win_in":    "Win_in.txt",
    "win_out":   "Win_out.txt",
}


# GRILLES
P_mbar_list = [1000, 1500, 2000, 2500, 3000, 4000]  # mbar
Lgas_list   = [100]                       # mm
p0_list     = [28]            # MeV/c
dpFWHM_list = [0.002]                      # %
nEvents = 20000
WindowRadius_list = [60]  # mm
gasMat_list = ["CH4_gas"]
T_K = 293.15
sigX, sigY, sigXp, sigYp = 5.0, 5.0, 0.0, 0.0

# Fenêtre
MATERIALS: Dict[str, Optional[List[float]]] = {
    # "KAPTON":   [0.25, 0.5,0.4], #50um, 1mm, 7.5 um
    # "MYLAR":     [0.25, 0.5,0.2], #50um, 1mm, 7.5 um
    # "Al":  [0.25],
    # "Ti":  [0.254],
    # "Havar":  [0.25],
    #C2H4
    "C2H4": [0.01, 1.0, 0.5],
    
}
G4BL_MATERIAL_NAME: Dict[str, str] = {
    # "KAPTON": "KAPTON",
    # "MYLAR": "MYLAR",
    # "Al": "Al",
    # "Ti": "Ti",
    # "Havar": "Havar",
    "C2H4": "C2H4",
}
WINDOW_MAT_KEYS = ["C2H4"]  

FORCE_RERUN = True
VERBOSE_FAIL = True



# ASCII helpers

def unique_event_ids(path: Path) -> set[int]:
    """Read G4BL asciiExtended and return unique EventID set."""
    event_col = None
    ev: set[int] = set()
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                if "EventID" in line:
                    headers = line.strip("#").split()
                    if "EventID" in headers:
                        event_col = headers.index("EventID")
                continue
            cols = line.split()
            if event_col is None:
                raise RuntimeError(f"EventID column not found in {path}")
            ev.add(int(float(cols[event_col])))
    return ev


def reached_sets_for_planes(outdir: Path) -> List[set[int]]:
    """Lit les fichiers ScanZ_*.txt et retourne une liste de sets d'EventID pour chaque plan de scan."""
    reached: List[set[int]] = [set() for _ in range(NSCAN)]
    for i, fn in enumerate(SCAN_FILES):
        p = outdir / fn
        if p.exists():
            reached[i] = unique_event_ids(p)
    return reached


def last_scan_index_per_event(reached: List[set[int]]) -> Dict[int, int]:
    """À partir de la liste de sets d'EventID par plan de scan, retourne un dict EventID -> dernier index de scan atteint."""
    all_ev = set().union(*reached) if reached else set()
    last_idx: Dict[int, int] = {}
    for ev in all_ev:
        for i in range(NSCAN - 1, -1, -1):
            if ev in reached[i]:
                last_idx[ev] = i
                break
    return last_idx


# def estimate_stop_z_mm(L_gas: float, last_idx: int) -> float:
#     dz = L_gas / NSCAN
#     return (last_idx + 1) * dz
def estimate_stop_z_mm(z_gas0: float, L_gas: float, last_idx: int) -> float:
    """Estime la position d'arrêt en z (mm) à partir du dernier index de scan atteint."""
    dz = L_gas / NSCAN
    return z_gas0 + (last_idx + 0.5) * dz

def safe_float(x: Optional[float]) -> str:
    """Formatte un float avec 6 chiffres significatifs, ou retourne une chaîne vide si None."""
    return "" if x is None else f"{x:.6g}"


def window_grid() -> List[Tuple[str, float, str]]:
    """Construit la grille des fenêtres à partir des matériaux et épaisseurs configurés."""
    grid = []
    for key in WINDOW_MAT_KEYS:
        th_list = MATERIALS.get(key)
        if not th_list:
            continue
        g4name = G4BL_MATERIAL_NAME.get(key)
        if not g4name:
            continue
        for t in th_list:
            if t is None or t <= 0:
                continue
            grid.append((key, float(t), g4name))
    return grid


WIN_GRID = window_grid()
if not WIN_GRID:
    raise RuntimeError("No window materials/thicknesses configured.")


def read_diag_counts(outdir: Path) -> Dict[str, int]:
    """Counts unique EventID for each diag file (0 if missing)."""
    out: Dict[str, int] = {}
    for k, fname in DIAG_FILES.items():
        p = outdir / fname
        if p.exists():
            try:
                out[k] = len(unique_event_ids(p))
            except Exception:
                out[k] = 0
        else:
            out[k] = 0
    return out


def ratio(numer: int, denom: int) -> Optional[float]:
    """Calcule le ratio numer/denom, retourne None si denom <= 0."""
    if denom <= 0:
        return None
    return numer / denom



#Main :scan + sauvegarde
results_csv = BASE_OUT / "results.csv"
write_header = not results_csv.exists()

fieldnames = [
    "timestamp", "tag", "outdir",
    "P_mbar", "T_K",
    "p0", "dpFWHM_percent",
    "L_gas",
    "sigX", "sigY", "sigXp", "sigYp",
    "win_key", "WindowMaterial", "L_Window", "GasMaterial",

    "N_in", "N_out", "frac_stop",
    "stop_mean", "stop_median",

    "Nveto_in", "Nveto_out",
    "Nscint_in", "Nscint_out",
    "radius",
    "Nwin_in", "Nwin_out",

    # transmissions
    "Tveto", "Tscint", "Twin_total", "rendement",
]

with open(results_csv, "a", newline="") as fcsv:
    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for (P_mbar, p0, Lgas, dpFWHM_percent, WindowRadius, gasMat, (win_key, L_Window, WindowMaterial)) in product(
        P_mbar_list, p0_list, Lgas_list, dpFWHM_list, WindowRadius_list, gasMat_list, WIN_GRID
    ):
        tag = (
            f"P{P_mbar}_p{p0}_dP{dpFWHM_percent:g}%_L{Lgas}Radius{WindowRadius}mm"
            f"_gas{gasMat}_W{win_key}-{WindowMaterial}_t{L_Window:.6f}mm"
        )

	
        outdir = BASE_OUT / tag
        outdir.mkdir(parents=True, exist_ok=True)

        zin = outdir / "Zin.txt"
        zout = outdir / "Zout.txt"

        # Run or reuse
        if (not FORCE_RERUN) and zin.exists() and zout.exists():
            pass
        else:
            cmd = [
                str(G4BL),
                str(G4BL_FILE),
                f"P_mbar={float(P_mbar)}",
                f"T_K={float(T_K)}",
                f"nEvents={int(nEvents)}",
                f"p0={float(p0)}",
                f"dpFWHM_percent={float(dpFWHM_percent)}",
                f"sigX={float(sigX)}",
                f"sigY={float(sigY)}",
                f"sigXp={float(sigXp)}",
                f"sigYp={float(sigYp)}",
                f"L_gas={float(Lgas)}",
                f"WindowRadius={float(WindowRadius)}",
                f"WindowMaterial={WindowMaterial}",
                f"L_Window={float(L_Window)}",
                f"GasMaterial={gasMat}",
                f"outDir={str(outdir)}",
            ]

            print(f"\nRUN: {tag}")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print("  g4bl failed -> skipping")
                if VERBOSE_FAIL:
                    print("\n".join(r.stdout.splitlines()[-10:]))
                    print("\n".join(r.stderr.splitlines()[-10:]))
                continue

        if not zin.exists() or not zout.exists():
            print(f"  Missing Zin/Zout in {outdir} -> skipping")
            continue

        # Gas entry/exit
        E_in = unique_event_ids(zin)
        E_out = unique_event_ids(zout)
        N_in = len(E_in)
        N_out = len(E_out)
        if N_in == 0:
            print("  No entries at Zin -> skipping")
            continue

        frac_stop = 1.0 - (N_out / N_in)

        # Stop-z estimate from scan planes (uniquement pour ceux qui entrent et ne sortent pas)
        reached = reached_sets_for_planes(outdir)
        last_idx = last_scan_index_per_event(reached)
        # stop_z = [
        #     estimate_stop_z_mm(float(Lgas), idx)
        #     for ev, idx in last_idx.items()
        #     if (ev in E_in and ev not in E_out)
        # ]
        stop_z = [
            estimate_stop_z_mm(0.0, float(Lgas), idx)
            for ev, idx in last_idx.items()
            if (ev in E_in and ev not in E_out)
        ]

        stop_mean = stats.mean(stop_z) if stop_z else None
        stop_median = stats.median(stop_z) if stop_z else None

        # Diagnostics
        diag = read_diag_counts(outdir)
        Nveto_in = diag["veto_in"]
        Nveto_out = diag["veto_out"]
        Nscint_in = diag["scint_in"]
        Nscint_out = diag["scint_out"]
        Nwin_in = diag["win_in"]
        Nwin_out = diag["win_out"]

        Tveto = ratio(Nveto_out, Nveto_in)
        Tscint = ratio(Nscint_out, Nscint_in)
        Twin_total = ratio(Nwin_out, Nwin_in)
        rendement = Twin_total * frac_stop

        print(
            f"  in={N_in} out={N_out} "
            f"frac_stop={frac_stop:.4f} "
            f"rendement={fmt(rendement)} "
            f"radius={WindowRadius}mm "
            f"stop_mean={fmt(stop_mean)} "
            f"stop_median={fmt(stop_median)} "
            f"| Tveto={fmt(Tveto)} "
            f"Tscint={fmt(Tscint)} "
            f"Twin={fmt(Twin_total)}"
        )

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tag": tag,
            "outdir": str(outdir),
            "P_mbar": P_mbar,
            "T_K": T_K,
            "p0": p0,
            "dpFWHM_percent": dpFWHM_percent,
            "L_gas": Lgas,
            "sigX": sigX,
            "sigY": sigY,
            "sigXp": sigXp,
            "sigYp": sigYp,
            "win_key": win_key,
            "WindowMaterial": WindowMaterial,
            "L_Window": L_Window,
            "GasMaterial": gasMat,
            "radius": WindowRadius,

            "N_in": N_in,
            "N_out": N_out,
            "frac_stop": frac_stop,
            "stop_mean": stop_mean,
            "stop_median": stop_median,

            "Nveto_in": Nveto_in,
            "Nveto_out": Nveto_out,
            "Nscint_in": Nscint_in,
            "Nscint_out": Nscint_out,
            "Nwin_in": Nwin_in,
            "Nwin_out": Nwin_out,

            "Tveto": Tveto,
            "Tscint": Tscint,
            "Twin_total": Twin_total,
            "rendement": rendement,
        }

        writer.writerow(row)
        fcsv.flush()