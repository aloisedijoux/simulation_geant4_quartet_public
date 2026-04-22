import os
import re
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend non-interactif
import matplotlib.pyplot as plt


# PARAMÈTRE : répertoire racine contenant tous les dossiers

racine = "/home/aloiselkb/g4bl_simu/test22042026/"  # ex: "/home/aloiselkb/g4bl_simu/Madalina_conf/2cmcell/"
filtre = None  # ex: "P1000" ou "t0.008000mm"


# MAPPING scan_index → z (mm)

L_gas   = 200.0   # mm
Nscan   = 400 
dz_scan = L_gas / Nscan  # 0.5 mm

def idx_to_z(i):
    return (i + 0.5) * dz_scan


# Format colonnes G4Beamline (0-indexé) :
# #x y z Px Py Pz t PDGid EventID TrackID ParentID Weight
#  0 1 2  3  4  5 6   7      8      9       10      11
#  Bx By Bz Ex Ey Ez ProperTime PathLength PolX PolY PolZ
#  12 13 14 15 16 17    18          19      20   21   22
#  InitX InitY InitZ InitT InitKE
#  23    24    25    26    27

COL_PX  = 3
COL_PY  = 4
COL_PZ  = 5
COL_PDG = 7
COL_ID  = 8

# Masse du muon (MeV/c²)
M_MU = 105.6583755

# PDGid muons
MUON_PDGIDS = {13, -13}


# Collecte des dossiers à traiter

def est_dossier_valide(path):
    for f in os.listdir(path):
        if f.startswith("ScanZ_") and f.endswith(".txt"):
            return True
    return False

dossiers = []
for nom in sorted(os.listdir(racine)):
    if nom.startswith("output_"):
        continue
    chemin = os.path.join(racine, nom)
    if not os.path.isdir(chemin):
        continue
    if filtre and filtre not in nom:
        continue
    if not est_dossier_valide(chemin):
        continue
    dossiers.append(chemin)

print(f"Nombre de dossiers à traiter : {len(dossiers)}")
for d in dossiers:
    print(f"  {os.path.basename(d)}")



# Helpers


def creer_output_dir(dossier):
    nom_config = os.path.basename(dossier.rstrip('/'))
    output_dir = os.path.join(os.path.dirname(dossier), f"output_{nom_config}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def savefig(fig, output_dir, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path


def cfg_label(nom_config):
    """Extrait P, p0, t depuis le nom du dossier pour les titres."""
    m = re.search(r"P(?P<P>[\d.]+)_p(?P<p0>[\d.]+).*_t(?P<t>[\d.]+)mm", nom_config)
    return f"P = {m.group('P')}, p0 = {m.group('p0')}, t = {m.group('t')} mm" if m else nom_config



# Étape 1 : comptage lignes ScanZ + fichiers spéciaux


def compter_lignes(dossier, log):
    scan_indices, line_counts = [], []

    for filename in os.listdir(dossier):
        if filename.startswith("ScanZ_") and filename.endswith(".txt"):
            match = re.search(r'ScanZ_(\d+)\.txt', filename)
            if match:
                scan_index = int(match.group(1))
                with open(os.path.join(dossier, filename), 'r') as f:
                    num_lines = sum(1 for _ in f)
                log(f"  {filename}: {num_lines} lignes")
                scan_indices.append(scan_index)
                line_counts.append(num_lines)

    sorted_data  = sorted(zip(scan_indices, line_counts))
    scan_indices, line_counts = zip(*sorted_data)

    special_counts = {}
    for nom in ["Win_in.txt", "Win_out.txt", "Zout.txt"]:
        p = os.path.join(dossier, nom)
        if os.path.isfile(p):
            with open(p, 'r') as fh:
                special_counts[nom] = sum(1 for _ in fh)
            log(f"  {nom}: {special_counts[nom]} lignes")
        else:
            special_counts[nom] = 0
            log(f"  {nom}: introuvable")

    return scan_indices, line_counts, special_counts


def fig_lignes(scan_indices, line_counts, special_counts, output_dir):
    z_vals = [idx_to_z(i) for i in scan_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(z_vals, line_counts, width=dz_scan)
    ax.set_yscale('log')
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('Nombre de lignes')
    ax.set_title('Nombre de lignes par fichier ScanZ')
    ax.grid(True, alpha=0.3)

    x_vals = [-2 * dz_scan, -dz_scan] + z_vals + [z_vals[-1] + dz_scan]
    y_vals = (
        [special_counts["Win_in.txt"], special_counts["Win_out.txt"]]
        + list(line_counts)
        + [special_counts["Zout.txt"]]
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x_vals, y_vals, width=dz_scan)
    ax.set_yscale("log")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Nombre de lignes")
    ax.set_title("Nombre de lignes (Win_in, Win_out, ScanZ, Zout)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()



# Étape 2 : matrice EventID


def construire_matrice(dossier):
    event_ids_par_scan = {}
    for filename in os.listdir(dossier):
        if filename.startswith("ScanZ_") and filename.endswith(".txt"):
            match = re.search(r'ScanZ_(\d+)\.txt', filename)
            if match:
                scan_index = int(match.group(1))
                event_ids  = set()
                with open(os.path.join(dossier, filename), 'r') as f:
                    for line in f:
                        if line.startswith('#'):
                            continue
                        parts = line.split()
                        if len(parts) > COL_ID:
                            try:
                                event_ids.add(int(parts[COL_ID]))
                            except ValueError:
                                pass
                event_ids_par_scan[scan_index] = event_ids

    scan_list  = sorted(event_ids_par_scan.keys())
    all_events = sorted(set().union(*event_ids_par_scan.values()))

    df = pd.DataFrame(False, index=scan_list, columns=all_events)
    for s in scan_list:
        df.loc[s, list(event_ids_par_scan[s])] = True
    df = df.astype(int)
    df.index.name   = "scan_index"
    df.columns.name = "eventID"
    return df



# Étape 3 : analyse zeros


def analyser_zeros(df_events, log):
    zeros = df_events[df_events == 0].stack().reset_index()
    zeros.columns = ['scan_index', 'eventID', 'value']
    log(f"  Nombre total de 0 dans la matrice : {len(zeros)}")

    zeros_excl_scan0 = zeros[zeros['scan_index'] != 0]
    zeros_excl = zeros_excl_scan0[
        ~zeros_excl_scan0['eventID'].isin(
            zeros[zeros['scan_index'] == 0]['eventID']
        )
    ]
    log(f"  Nb de 0 hors scan_index=0 (events présents au moins une fois) : {len(zeros_excl)}")

    event_id_zeros = zeros_excl['eventID'].unique()
    log(f"  EventID uniques concernés ({len(event_id_zeros)}) : {sorted(event_id_zeros)}")

    for event_id in sorted(event_id_zeros):
        scans_avec = df_events[df_events[event_id] == 1].index
        scans_sans = df_events[df_events[event_id] == 0].index
        if len(scans_avec) == 0:
            log(f"  EventID {event_id}: absent de tous les scans.")
            continue
        last_avec  = scans_avec.max()
        first_sans = scans_sans[scans_sans > last_avec]
        first_sans = first_sans.min() if len(first_sans) > 0 else None
        log(
            f"  EventID {event_id:>6d}: "
            f"dernier scan présent = {last_avec} (z={idx_to_z(last_avec):.3f} mm), "
            f"premier scan absent  = {first_sans}"
            + (f" (z={idx_to_z(first_sans):.3f} mm)" if first_sans is not None else "")
        )
    return event_id_zeros



# Étape 4 : arrêts définitifs + figure


def analyser_arrets(df_events, output_dir, log):
    last_scan_global = df_events.index.max()
    last_scan_par_event = df_events.apply(
        lambda col: col[col == 1].index.max() if (col == 1).any() else None
    )
    filtre_arr = last_scan_par_event[last_scan_par_event < last_scan_global]
    nb_stops = filtre_arr.value_counts().sort_index()

    nb_stops_z = nb_stops.copy()
    nb_stops_z.index = nb_stops_z.index.map(idx_to_z)

    log("  Arrêts définitifs par z (mm) (hors dernier scan) :")
    log("  " + nb_stops_z.to_string().replace("\n", "\n  "))

    nom_config = os.path.basename(output_dir).replace("output_", "")
    cfg_txt = cfg_label(nom_config)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(nb_stops_z.index, nb_stops_z.values, width=dz_scan)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Nb events arrêtés")
    ax.set_xlim(0, L_gas)
    ax.set_title(f"Arrêt définitif des events par z (hors dernier scan)\n{cfg_txt}")
    ax.grid(True, alpha=0.3)
    savefig(fig, output_dir, "fig3_arrets_definitifs.png")



# Étape 5 : EventID max + nb events par scan


def analyser_eventid_max(df_events, output_dir, log):
    df_num = df_events.copy()
    df_num.columns = df_num.columns.astype(int)
    df_num = df_num.reindex(sorted(df_num.columns), axis=1)
    cols = df_num.columns.to_numpy()
    arr  = df_num.to_numpy()

    event_id_max_par_scan = pd.Series(
        (arr * cols).max(axis=1), index=df_num.index
    ).astype(int).sort_index()

    z_vals = event_id_max_par_scan.index.map(idx_to_z)

    log(f"  EventID max : nunique={event_id_max_par_scan.nunique()}  "
        f"min={event_id_max_par_scan.min()}  max={event_id_max_par_scan.max()}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(z_vals, event_id_max_par_scan.values, width=dz_scan)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("EventID max")
    ax.set_xlim(0, L_gas)
    ax.set_title("EventID max par scan (barres)")
    ax.grid(True, alpha=0.3)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z_vals, event_id_max_par_scan.values)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("EventID max présent")
    ax.set_xlim(0, L_gas)
    ax.set_title("EventID max par scan (courbe)")
    ax.grid(True, alpha=0.3)

    nb_events_par_scan = df_events.sum(axis=1).sort_index()
    z_vals2 = nb_events_par_scan.index.map(idx_to_z)

    log(f"  Nb events/scan : min={nb_events_par_scan.min()}  max={nb_events_par_scan.max()}  "
        f"médiane={nb_events_par_scan.median()}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z_vals2, nb_events_par_scan.values)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Nb EventID présents")
    ax.set_xlim(0, L_gas)
    ax.set_title("Nombre d'EventID présents par scan")
    ax.grid(True, alpha=0.3)



# Étape 6 : lecture Px/Py/Pz par scan — muons uniquement
#
#   KE relativiste = sqrt(|p|² + M_MU²) - M_MU   [MeV]
#   |p| = sqrt(Px² + Py² + Pz²)                   [MeV/c]


def lire_ke_p_par_scan(dossier, log):
    """
    Lit Px, Py, Pz et PDGid dans chaque ScanZ_*.txt.
    Filtre PDGid = ±13 (muons), calcule KE relativiste.
    Retourne dict  scan_index → {'ke': array, 'p': array}.
    """
    data    = {}
    n_total = 0
    n_muons = 0

    for filename in sorted(os.listdir(dossier)):
        if not (filename.startswith("ScanZ_") and filename.endswith(".txt")):
            continue
        match = re.search(r'ScanZ_(\d+)\.txt', filename)
        if not match:
            continue
        idx = int(match.group(1))

        ke_list, p_list = [], []
        with open(os.path.join(dossier, filename), 'r') as fh:
            for line in fh:
                if line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) <= max(COL_PZ, COL_PDG):
                    continue
                try:
                    pdg = int(parts[COL_PDG])
                except ValueError:
                    continue
                n_total += 1
                if pdg not in MUON_PDGIDS:
                    continue
                n_muons += 1
                try:
                    px = float(parts[COL_PX])
                    py = float(parts[COL_PY])
                    pz = float(parts[COL_PZ])
                except ValueError:
                    continue
                p_mag = np.sqrt(px**2 + py**2 + pz**2)
                ke    = np.sqrt(p_mag**2 + M_MU**2) - M_MU
                ke_list.append(ke)
                p_list.append(p_mag)

        if ke_list:
            data[idx] = {
                'ke': np.array(ke_list),
                'p' : np.array(p_list),
            }

    frac = 100 * n_muons / n_total if n_total > 0 else 0
    log(f"  Hits totaux : {n_total}  |  muons (±13) : {n_muons} ({frac:.1f}%)")
    log(f"  Scans avec données muons : {len(data)}")
    return data



# Étape 7 : -dE/dx vs z  +  <KE> ± σ vs z


def fig_dEdx(ke_p_data, output_dir, log):
    """
    -dE/dx = -d<KE>/dz estimé par np.gradient sur les scans consécutifs.
    Produit :
      fig7a_dEdx_vs_z.png
      fig7b_KE_moyen_vs_z.png
    """
    scan_list = sorted(ke_p_data.keys())
    z_vals    = np.array([idx_to_z(i) for i in scan_list])
    ke_mean   = np.array([ke_p_data[i]['ke'].mean() for i in scan_list])
    ke_std    = np.array([ke_p_data[i]['ke'].std()  for i in scan_list])

    # -dE/dx  (MeV/mm) — positif = perte d'énergie
    dEdx = -np.gradient(ke_mean, z_vals)

    log(f"  <KE> muons : min={ke_mean.min():.2f}  max={ke_mean.max():.2f} MeV")
    log(f"  -dE/dx     : min={dEdx.min():.4f}  max={dEdx.max():.4f} MeV/mm")

    nom_config = os.path.basename(output_dir).replace("output_", "")
    cfg_txt    = cfg_label(nom_config)

    # ── Fig 7a : -dE/dx vs z ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z_vals, dEdx, lw=1.5, color='steelblue')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel("z (mm)")
    ax.set_ylabel(r"$-dE/dx$ (MeV/mm)")
    ax.set_xlim(0, L_gas)
    ax.set_title(
        f"Perte d'énergie linéique moyenne des muons $-dE/dx$ vs z\n{cfg_txt}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, output_dir, "fig7a_dEdx_vs_z.png")

    # ── Fig 7b : <KE> ± σ vs z ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(z_vals, ke_mean - ke_std, ke_mean + ke_std,
                    alpha=0.25, color='darkorange', label=r'$\pm 1\sigma$')
    ax.plot(z_vals, ke_mean, lw=1.5, color='darkorange',
            label=r'$\langle KE \rangle$')
    ax.set_xlabel("z (mm)")
    ax.set_ylabel(r"$\langle KE \rangle$ (MeV)")
    ax.set_xlim(0, L_gas)
    ax.set_title(f"Énergie cinétique moyenne des muons par scan\n{cfg_txt}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, output_dir, "fig7b_KE_moyen_vs_z.png")



# Étape 8 : KE vs |p|  (scatter + profil médian)


def fig_KE_vs_p(ke_p_data, output_dir, log):
    """
    Scatter KE vs |p| pour tous les hits muons (décimé si > 200 k points).
    Superpose la dispersion relativiste théorique KE = sqrt(p² + M_MU²) - M_MU.
    Produit :
      fig8a_KE_vs_p_scatter.png
      fig8b_KE_vs_p_profil.png
    """
    all_ke = np.concatenate([ke_p_data[i]['ke'] for i in sorted(ke_p_data)])
    all_p  = np.concatenate([ke_p_data[i]['p']  for i in sorted(ke_p_data)])
    log(f"  Total points muons pour KE/p : {len(all_ke)}")

    # Courbe théorique muon relativiste (relation de dispersion)
    p_th  = np.linspace(0, all_p.max() * 1.05, 600)
    ke_th = np.sqrt(p_th**2 + M_MU**2) - M_MU

    nom_config = os.path.basename(output_dir).replace("output_", "")
    cfg_txt    = cfg_label(nom_config)

    # ── Fig 8a : scatter (décimé si nécessaire) ──
    MAX_PTS = 200_000
    ke_sc, p_sc = all_ke, all_p
    if len(all_ke) > MAX_PTS:
        rng      = np.random.default_rng(42)
        idx_dec  = rng.choice(len(all_ke), MAX_PTS, replace=False)
        ke_sc    = all_ke[idx_dec]
        p_sc     = all_p[idx_dec]
        log(f"  → scatter décimé à {MAX_PTS} points")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(p_sc, ke_sc, s=1, alpha=0.15, color='royalblue',
               label='muons (données)')
    ax.plot(p_th, ke_th, 'r-', lw=2,
            label=r'$KE = \sqrt{p^2 + m_\mu^2} - m_\mu$')
    ax.set_xlabel(r"$|p|$ (MeV/c)")
    ax.set_ylabel(r"$KE$ (MeV)")
    ax.set_title(f"Énergie cinétique vs impulsion — muons\n{cfg_txt}")
    ax.legend(markerscale=5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    #savefig(fig, output_dir, "fig8a_KE_vs_p_scatter.png")

    # ── Fig 8b : profil médian + IQR par bin d'impulsion ──
    p_min, p_max = all_p.min(), all_p.max()
    n_bins  = max(10, min(80, int((p_max - p_min) / 5)))  # bins ~5 MeV/c
    bins    = np.linspace(p_min, p_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    medians, q25, q75 = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (all_p >= lo) & (all_p < hi)
        if mask.sum() < 5:
            medians.append(np.nan)
            q25.append(np.nan)
            q75.append(np.nan)
        else:
            v = all_ke[mask]
            medians.append(np.median(v))
            q25.append(np.percentile(v, 25))
            q75.append(np.percentile(v, 75))

    medians = np.array(medians)
    q25     = np.array(q25)
    q75     = np.array(q75)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(centers, q25, q75,
                    alpha=0.3, color='royalblue', label='IQR [25–75 %]')
    ax.plot(centers, medians, 'o-', ms=3, lw=1.5,
            color='royalblue', label='médiane KE')
    ax.plot(p_th, ke_th, 'r-', lw=2,
            label=r'$KE = \sqrt{p^2 + m_\mu^2} - m_\mu$')
    ax.set_xlabel(r"$|p|$ (MeV/c)")
    ax.set_ylabel(r"$KE$ (MeV)")
    ax.set_title(f"Profil médian KE vs |p| — muons (tous scans)\n{cfg_txt}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    #savefig(fig, output_dir, "fig8b_KE_vs_p_profil.png")



# BOUCLE PRINCIPALE

print("Fonctions d'analyse chargées.")

erreurs = []

for i, dossier in enumerate(dossiers):
    nom_config = os.path.basename(dossier)
    print(f"\n[{i+1}/{len(dossiers)}] ── {nom_config}")

    try:
        output_dir = creer_output_dir(dossier)
        log_path   = os.path.join(output_dir, "log.txt")
        log_file   = open(log_path, 'w')

        def log(msg=""):
            print(msg)
            log_file.write(str(msg) + "\n")

        log(f" Analyse de : {dossier} ")
        log(f"Dossier de sortie : {output_dir}")
        log(f"L_gas={L_gas} mm, Nscan={Nscan}, dz_scan={dz_scan} mm")
        log(f"Filtre PDGid muons : {MUON_PDGIDS}")
        log()

        # ── 1. Lignes ──
        log("--- Comptage des lignes ---")
        scan_indices, line_counts, special_counts = compter_lignes(dossier, log)
        fig_lignes(scan_indices, line_counts, special_counts, output_dir)
        log()

        # ── 2. Matrice EventID ──
        log("--- Construction de la matrice EventID ---")
        df_events = construire_matrice(dossier)
        log(f"  Shape : {df_events.shape}  (scans x eventIDs)")
        log()

        # ── 3. Zéros ──
        log("--- Analyse des zéros ---")
        analyser_zeros(df_events, log)
        log()

        # ── 4. Arrêts définitifs ──
        log("--- Arrêts définitifs ---")
        analyser_arrets(df_events, output_dir, log)
        log()

        # ── 5. EventID max ──
        log("--- EventID max & nb events par scan ---")
        analyser_eventid_max(df_events, output_dir, log)
        log()

        # ── 6–8. KE / impulsion / -dE/dx (muons ±13) ──
        log("--- Lecture KE et impulsion (muons PDGid=±13) par scan ---")
        ke_p_data = lire_ke_p_par_scan(dossier, log)
        log()

        if ke_p_data:
            log("--- -dE/dx et KE moyen vs z ---")
            fig_dEdx(ke_p_data, output_dir, log)
            log()

            log("--- KE vs |p| ---")
            fig_KE_vs_p(ke_p_data, output_dir, log)
            log()
        else:
            log("  Aucun muon (PDGid=±13) trouvé — figures 7 et 8 ignorées.")
            log()

        log(" Terminé ")
        log_file.close()

        fichiers = sorted(os.listdir(output_dir))
        print(f"  → {len(fichiers)} fichiers dans {output_dir}")

    except Exception as e:
        msg = f"ERREUR sur {nom_config} : {e}"
        print(msg)
        traceback.print_exc()
        erreurs.append((nom_config, str(e)))
        try:
            log_file.write(msg + "\n")
            log_file.close()
        except Exception:
            pass


print(f"Analyse terminée : {len(dossiers) - len(erreurs)}/{len(dossiers)} dossiers OK")
if erreurs:
    print("Dossiers en erreur :")
    for nom, err in erreurs:
        print(f"  {nom} → {err}")
