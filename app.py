import streamlit as st
import pandas as pd
import tempfile
import subprocess
import re
import io
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Gaussian SDF Dedupe App",
    layout="wide",
)

st.title("Gaussian SDF Dedupe App")
st.caption("Ver. 1.0")
st.write("Convert Gaussian .log files to SDF, merge conformers, and filter structures by RMSD.")
st.info(
    "Use symmetry-aware RMSD for symmetric structures such as para-substituted benzenes. "
    "For ordinary asymmetric structures, direct alignment RMSD can be used."
)

# session state initialization
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

if "result_payload" not in st.session_state:
    st.session_state.result_payload = None


# =========================
# Regex / constants
# =========================
SCF_RE = re.compile(r"SCF Done:\s+E\([RU]?\w+\)\s+=\s+(-?\d+\.\d+)")
ARCHIVE_HF_RE = re.compile(r"\|HF=(-?\d+\.\d+)")
GIBBS_KEY = "Sum of electronic and thermal Free Energies"
NORMAL_TERM_KEY = "Normal termination of Gaussian"
ENERGY_HARTREE_TO_KCAL = 627.509474


# =========================
# Helper functions
# =========================
def check_obabel_available():
    try:
        result = subprocess.run(
            ["obabel", "-H"],
            capture_output=True,
            text=True,
            timeout=20
        )
        if result.returncode == 0:
            return True, "Open Babel detected."
        return False, f"Open Babel returned non-zero exit status.\nSTDERR:\n{result.stderr}"
    except FileNotFoundError:
        return False, "Open Babel ('obabel') was not found."
    except Exception as e:
        return False, f"Failed to run obabel: {e}"


def save_uploaded_file(uploaded_file, output_path: Path):
    output_path.write_bytes(uploaded_file.getbuffer())


def extract_last_scf_energy(logfile: Path):
    last = None
    archive_last = None

    with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = SCF_RE.search(line)
            if m:
                last = float(m.group(1))

            for m2 in ARCHIVE_HF_RE.finditer(line):
                archive_last = float(m2.group(1))

    if last is not None:
        return last
    return archive_last


def extract_gibbs_energy(logfile: Path):
    found = None
    with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if GIBBS_KEY in line:
                try:
                    found = float(line.split()[-1])
                except Exception:
                    pass
    return found


def check_normal_termination(logfile: Path):
    found = False
    with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if NORMAL_TERM_KEY in line:
                found = True
    return found


def convert_log_to_sdf(log_path: Path, sdf_path: Path):
    try:
        result = subprocess.run(
            ["obabel", str(log_path), "-O", str(sdf_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        success = (result.returncode == 0) and sdf_path.exists() and sdf_path.stat().st_size > 0
        return success, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def read_first_mol_from_sdf(sdf_path: Path):
    try:
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            return None
        return mols[0]
    except Exception:
        return None


def set_energy_property_to_sdf(
    sdf_path: Path,
    source_name: str,
    energy_value: float,
    energy_type_label: str,
    normal_termination: bool
):
    mol = read_first_mol_from_sdf(sdf_path)
    if mol is None:
        return False, "RDKit failed to read the generated SDF."

    mol.SetProp("SourceFile", source_name)
    mol.SetProp("EnergyType", energy_type_label)
    mol.SetProp(energy_type_label, str(energy_value))
    mol.SetProp("NormalTermination", "True" if normal_termination else "False")

    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()
    return True, ""


def load_molecules_from_sdfs(sdf_paths):
    mols = []
    failed = []
    for sdf_path in sdf_paths:
        mol = read_first_mol_from_sdf(sdf_path)
        if mol is None:
            failed.append(str(sdf_path.name))
        else:
            mols.append(mol)
    return mols, failed


def get_energy(mol, energy_type_label):
    if mol.HasProp(energy_type_label):
        try:
            return float(mol.GetProp(energy_type_label))
        except Exception:
            return None
    return None


def write_sdf_bytes(mols):
    sio = io.StringIO()
    writer = Chem.SDWriter(sio)
    for mol in mols:
        writer.write(mol)
    writer.close()
    text = sio.getvalue()
    return text.encode("utf-8")


def calculate_rmsd(mol_a, mol_b, method="GetBestRMS", remove_hs_for_rmsd=True):
    try:
        a = Chem.RemoveHs(mol_a) if remove_hs_for_rmsd else mol_a
        b = Chem.RemoveHs(mol_b) if remove_hs_for_rmsd else mol_b

        if method == "AlignMol":
            rmsd = AllChem.AlignMol(a, b)
        else:
            rmsd = AllChem.GetBestRMS(a, b)
        return rmsd
    except Exception:
        return None


def deduplicate_molecules(
    mols,
    energy_type_label="SCF",
    rmsd_cutoff=0.20,
    remove_hs_for_rmsd=True,
    rmsd_method="GetBestRMS"
):
    prepared = []
    for idx, mol in enumerate(mols):
        energy = get_energy(mol, energy_type_label)
        source = mol.GetProp("SourceFile") if mol.HasProp("SourceFile") else f"mol_{idx+1}"
        normal_term = mol.GetProp("NormalTermination") if mol.HasProp("NormalTermination") else ""
        prepared.append({
            "mol": mol,
            "energy": energy,
            "source": source,
            "normal_termination": normal_term,
        })

    valid = [x for x in prepared if x["energy"] is not None]
    invalid = [x for x in prepared if x["energy"] is None]

    valid.sort(key=lambda x: x["energy"])

    kept = []
    kept_info = []
    summary_rows = []

    for rank, item in enumerate(valid, start=1):
        mol = item["mol"]

        is_dup = False
        duplicate_of = ""
        best_rmsd = None

        for kept_item in kept_info:
            kept_mol = kept_item["mol"]

            rmsd = calculate_rmsd(
                mol,
                kept_mol,
                method=rmsd_method,
                remove_hs_for_rmsd=remove_hs_for_rmsd
            )

            if rmsd is not None and rmsd < rmsd_cutoff:
                is_dup = True
                duplicate_of = kept_item["source"]
                best_rmsd = rmsd
                break

        if not is_dup:
            kept.append(mol)
            kept_info.append(item)

        summary_rows.append({
            "source_file": item["source"],
            "energy_type": energy_type_label,
            "energy_hartree": item["energy"],
            "relative_energy_kcal_mol": None,
            "rank_by_energy": rank,
            "status": "removed_as_duplicate" if is_dup else "kept",
            "duplicate_of": duplicate_of,
            "rmsd_to_representative": best_rmsd,
            "normal_termination": item["normal_termination"],
        })

    for item in invalid:
        summary_rows.append({
            "source_file": item["source"],
            "energy_type": energy_type_label,
            "energy_hartree": None,
            "relative_energy_kcal_mol": None,
            "rank_by_energy": None,
            "status": "energy_not_found_after_sdf_read",
            "duplicate_of": "",
            "rmsd_to_representative": None,
            "normal_termination": item["normal_termination"],
        })

    valid_energies = [row["energy_hartree"] for row in summary_rows if row["energy_hartree"] is not None]
    if valid_energies:
        e0 = min(valid_energies)
        for row in summary_rows:
            if row["energy_hartree"] is not None:
                row["relative_energy_kcal_mol"] = (row["energy_hartree"] - e0) * ENERGY_HARTREE_TO_KCAL

    return kept, summary_rows


def make_summary_csv_bytes(summary_rows):
    df = pd.DataFrame(summary_rows)
    csv_text = df.to_csv(index=False)
    return df, csv_text.encode("utf-8")


def render_results(payload):
    st.subheader("Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Uploaded log files", payload["uploaded_count"])
    col2.metric("Valid conformers", payload["valid_count"])
    col3.metric("Structures kept", payload["kept_count"])

    st.subheader("Summary table")
    st.dataframe(payload["display_df"], use_container_width=True)

    st.subheader("Download outputs")
    d1, d2, d3 = st.columns(3)

    d1.download_button(
        label="Download all_conformers.sdf",
        data=payload["all_sdf_bytes"],
        file_name="all_conformers.sdf",
        mime="chemical/x-mdl-sdfile"
    )

    d2.download_button(
        label="Download unique_conformers.sdf",
        data=payload["unique_sdf_bytes"],
        file_name="unique_conformers.sdf",
        mime="chemical/x-mdl-sdfile"
    )

    d3.download_button(
        label="Download summary.csv",
        data=payload["summary_csv_bytes"],
        file_name="summary.csv",
        mime="text/csv"
    )

    with st.expander("Show conversion details"):
        if payload["conversion_logs_df"] is not None and not payload["conversion_logs_df"].empty:
            st.dataframe(payload["conversion_logs_df"], use_container_width=True)
        else:
            st.write("No conversion issues were recorded.")


# =========================
# Sidebar settings
# =========================
with st.sidebar:
    st.header("Settings")

    energy_type_ui = st.radio(
        "Energy to extract",
        options=["SCF", "Free Energy"],
        index=0,
        help="Choose which energy to extract from Gaussian log files."
    )

    if energy_type_ui == "SCF":
        energy_type_label = "SCF"
    else:
        energy_type_label = "Free Energy"

    symmetry_mode = st.radio(
        "Symmetry handling",
        options=[
            "Symmetric structure present",
            "No symmetric structure concern",
        ],
        index=0,
        help="Use symmetry-aware RMSD for para-substituted benzene-like cases."
    )

    if symmetry_mode == "Symmetric structure present":
        rmsd_method = "GetBestRMS"
        st.caption("RMSD method: GetBestRMS (symmetry-aware)")
    else:
        rmsd_method = "AlignMol"
        st.caption("RMSD method: AlignMol (direct alignment)")

    rmsd_cutoff = st.number_input(
        "RMSD cutoff (Å)",
        min_value=0.01,
        max_value=10.00,
        value=0.20,
        step=0.01,
        format="%.2f",
        help="Conformers with RMSD below this cutoff are treated as duplicates."
    )

    remove_hs_for_rmsd = st.checkbox(
        "Use heavy-atom RMSD (remove H atoms for RMSD calculation)",
        value=True
    )

    if st.button("Clear results"):
        st.session_state.results_ready = False
        st.session_state.result_payload = None
        st.rerun()


# =========================
# Main UI
# =========================
uploaded_files = st.file_uploader(
    "Upload Gaussian .log files",
    type=["log"],
    accept_multiple_files=True
)

run_button = st.button("Run conversion and filtering", type="primary")


# =========================
# Run
# =========================
if run_button:
    if not uploaded_files:
        st.error("Please upload at least one Gaussian .log file.")
        st.stop()

    obabel_ok, obabel_msg = check_obabel_available()
    if not obabel_ok:
        st.error("Open Babel is not available.")
        st.code(obabel_msg)
        st.stop()

    st.success(obabel_msg)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        sdf_paths = []
        conversion_logs = []

        progress = st.progress(0)
        status_box = st.empty()

        total_files = len(uploaded_files)

        for i, uploaded_file in enumerate(uploaded_files, start=1):
            status_box.write(f"Processing {i}/{total_files}: `{uploaded_file.name}`")

            log_path = tmpdir / uploaded_file.name
            save_uploaded_file(uploaded_file, log_path)

            base_name = log_path.stem
            sdf_path = tmpdir / f"{base_name}.sdf"

            normal_term = check_normal_termination(log_path)

            if energy_type_label == "SCF":
                energy_value = extract_last_scf_energy(log_path)
            else:
                energy_value = extract_gibbs_energy(log_path)

            conv_success, conv_stdout, conv_stderr = convert_log_to_sdf(log_path, sdf_path)

            record = {
                "source_file": uploaded_file.name,
                "energy_type": energy_type_label,
                "energy_hartree": energy_value,
                "relative_energy_kcal_mol": None,
                "rank_by_energy": None,
                "status": "",
                "duplicate_of": "",
                "rmsd_to_representative": None,
                "normal_termination": normal_term,
                "obabel_stdout": conv_stdout,
                "obabel_stderr": conv_stderr,
            }

            if not conv_success:
                record["status"] = "conversion_failed"
                conversion_logs.append(record)
                progress.progress(i / total_files)
                continue

            if energy_value is None:
                record["status"] = "energy_not_found_in_log"
                conversion_logs.append(record)
                progress.progress(i / total_files)
                continue

            ok, msg = set_energy_property_to_sdf(
                sdf_path=sdf_path,
                source_name=uploaded_file.name,
                energy_value=energy_value,
                energy_type_label=energy_type_label,
                normal_termination=normal_term
            )

            if not ok:
                record["status"] = "sdf_property_write_failed"
                record["obabel_stderr"] = (record["obabel_stderr"] or "") + f"\n{msg}"
                conversion_logs.append(record)
                progress.progress(i / total_files)
                continue

            sdf_paths.append(sdf_path)
            progress.progress(i / total_files)

        status_box.write("Conversion step finished.")

        if not sdf_paths:
            st.error("No valid SDF files were generated.")
            if conversion_logs:
                st.subheader("Conversion log")
                st.dataframe(pd.DataFrame(conversion_logs), use_container_width=True)
            st.stop()

        mols, sdf_read_failed = load_molecules_from_sdfs(sdf_paths)

        if not mols:
            st.error("RDKit could not read any generated SDF files.")
            if conversion_logs:
                st.subheader("Conversion log")
                st.dataframe(pd.DataFrame(conversion_logs), use_container_width=True)
            st.stop()

        all_sdf_bytes = write_sdf_bytes(mols)

        kept_mols, result_summary_rows = deduplicate_molecules(
            mols=mols,
            energy_type_label=energy_type_label,
            rmsd_cutoff=rmsd_cutoff,
            remove_hs_for_rmsd=remove_hs_for_rmsd,
            rmsd_method=rmsd_method
        )

        unique_sdf_bytes = write_sdf_bytes(kept_mols)

        summary_rows = result_summary_rows[:]

        for row in conversion_logs:
            if row["status"] in {"conversion_failed", "energy_not_found_in_log", "sdf_property_write_failed"}:
                summary_rows.append({
                    "source_file": row["source_file"],
                    "energy_type": row["energy_type"],
                    "energy_hartree": row["energy_hartree"],
                    "relative_energy_kcal_mol": None,
                    "rank_by_energy": None,
                    "status": row["status"],
                    "duplicate_of": "",
                    "rmsd_to_representative": None,
                    "normal_termination": row["normal_termination"],
                })

        for name in sdf_read_failed:
            summary_rows.append({
                "source_file": name,
                "energy_type": energy_type_label,
                "energy_hartree": None,
                "relative_energy_kcal_mol": None,
                "rank_by_energy": None,
                "status": "rdkit_read_failed_after_conversion",
                "duplicate_of": "",
                "rmsd_to_representative": None,
                "normal_termination": "",
            })

        summary_df, summary_csv_bytes = make_summary_csv_bytes(summary_rows)

        display_df = summary_df.sort_values(
            by=["status", "rank_by_energy", "source_file"],
            na_position="last"
        ).reset_index(drop=True)

        conversion_logs_df = pd.DataFrame(conversion_logs) if conversion_logs else pd.DataFrame()

        st.session_state.result_payload = {
            "uploaded_count": len(uploaded_files),
            "valid_count": len(mols),
            "kept_count": len(kept_mols),
            "display_df": display_df,
            "all_sdf_bytes": all_sdf_bytes,
            "unique_sdf_bytes": unique_sdf_bytes,
            "summary_csv_bytes": summary_csv_bytes,
            "conversion_logs_df": conversion_logs_df,
        }
        st.session_state.results_ready = True

# render from session state
if st.session_state.results_ready and st.session_state.result_payload is not None:
    render_results(st.session_state.result_payload)
