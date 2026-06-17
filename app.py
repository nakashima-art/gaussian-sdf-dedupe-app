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
    page_title="Gaussian / SDF Dedupe App",
    layout="wide",
)

# =========================
# App metadata
# =========================
APP_VERSION = "1.1"

DEVELOPER_INFO = {
    "name": "Ken-ichi Nakashima",
    "affiliation_ja": "愛知学院大学 薬学部 薬用資源学講座",
    "affiliation_en": "Aichi-Gakuin University, School of Pharmacy, Laboratory of Natural Resources",
}

# Internal property names used in this app
APP_ENERGY_PROP = "AppEnergy"
APP_ENERGY_TYPE_PROP = "AppEnergyType"
APP_ENERGY_UNIT_PROP = "AppEnergyUnit"
APP_ENERGY_SOURCE_PROP = "AppEnergySource"
APP_SOURCE_FILE_PROP = "SourceFile"
APP_SOURCE_LABEL_PROP = "SourceLabel"
APP_INPUT_TYPE_PROP = "InputType"
APP_RECORD_INDEX_PROP = "RecordIndex"
APP_NORMAL_TERM_PROP = "NormalTermination"

# =========================
# Regex / constants
# =========================
SCF_RE = re.compile(r"SCF Done:\s+E\([RU]?\w+\)\s+=\s+(-?\d+\.\d+)")
ARCHIVE_HF_RE = re.compile(r"\|HF=(-?\d+\.\d+)")
FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")

GIBBS_KEY = "Sum of electronic and thermal Free Energies"
NORMAL_TERM_KEY = "Normal termination of Gaussian"

ENERGY_HARTREE_TO_KCAL = 627.509474
ENERGY_KJ_TO_KCAL = 1.0 / 4.184

COMMON_SDF_ENERGY_PROPS = [
    "Energy",
    "energy",
    "ENERGY",
    "MMFF94 Energy",
    "MMFF94_energy",
    "CONFLEX Energy",
    "CONFLEX_Energy",
    "Total Energy",
    "Final Energy",
    "E",
    "SCF",
    "Free Energy",
    "Gibbs Free Energy",
]

# =========================
# UI text
# =========================
UI_TEXT = {
    "en": {
        "language_selector": "Language / 言語",
        "title": "Gaussian / SDF Dedupe App",
        "caption": f"Ver. {APP_VERSION}",
        "description": "Convert Gaussian .log files to SDF, read conformer SDF files, and filter structures by RMSD.",
        "info": (
            "This app supports Gaussian .log files and conformer .sdf files (e.g., CONFLEX output). "
            "For SDF input, select the energy property and unit appropriately. "
            "If you mix Gaussian and SDF files in one run, energy ranking is only meaningful when the energy scales are comparable."
        ),
        "settings": "Settings",
        "developer_info": "Developer information",
        "developer_name": "Name",
        "developer_affiliation": "Affiliation",
        "log_energy_header": "Gaussian .log settings",
        "energy_to_extract": "Energy to extract from Gaussian .log",
        "scf": "SCF",
        "free_energy": "Free Energy",
        "sdf_energy_header": "SDF settings",
        "sdf_energy_mode": "SDF energy property",
        "auto_detect": "Auto-detect common energy property",
        "manual_property": "Specify property name manually",
        "sdf_property_name": "SDF property name",
        "sdf_property_hint": "Examples: Energy, MMFF94 Energy, CONFLEX Energy",
        "sdf_energy_unit": "SDF energy unit",
        "hartree": "Hartree",
        "kcal_mol": "kcal/mol",
        "kj_mol": "kJ/mol",
        "unknown_unit": "Unknown (do not calculate relative energy)",
        "symmetry_header": "RMSD settings",
        "symmetry_handling": "Symmetry handling",
        "symmetric_present": "Symmetric structure present",
        "no_symmetric_concern": "No symmetric structure concern",
        "rmsd_getbestrms": "RMSD method: GetBestRMS (symmetry-aware)",
        "rmsd_alignmol": "RMSD method: AlignMol (direct alignment)",
        "rmsd_cutoff": "RMSD cutoff (Å)",
        "rmsd_cutoff_help": "Conformers with RMSD below this cutoff are treated as duplicates.",
        "heavy_atom_rmsd": "Use heavy-atom RMSD (remove H atoms for RMSD calculation)",
        "clear_results": "Clear results",
        "upload_files": "Upload Gaussian .log and/or conformer .sdf files",
        "run_button": "Run processing and filtering",
        "need_file": "Please upload at least one .log or .sdf file.",
        "obabel_not_available": "Open Babel is not available.",
        "processing": "Processing",
        "conversion_finished": "File processing finished.",
        "no_valid_molecules": "No valid structures could be loaded.",
        "results": "Results",
        "uploaded_files_metric": "Uploaded files",
        "loaded_conformers_metric": "Loaded conformers",
        "kept_structures_metric": "Structures kept",
        "summary_table": "Summary table",
        "download_outputs": "Download outputs",
        "download_all": "Download all_conformers.sdf",
        "download_unique": "Download unique_conformers.sdf",
        "download_summary": "Download summary.csv",
        "show_details": "Show processing details",
        "no_details": "No processing issues were recorded.",
        "mixed_input_warning": (
            "Both Gaussian .log and SDF files are included. "
            "Representative selection is only meaningful when all uploaded energies are on a comparable scale."
        ),
        "status_kept": "kept",
        "status_removed_as_duplicate": "removed as duplicate",
        "status_energy_not_found": "energy not found",
        "status_conversion_failed": "conversion failed",
        "status_rdkit_read_failed": "RDKit read failed",
        "status_unsupported_file_type": "unsupported file type",
        "status_manual_property_not_found": "specified SDF property not found",
        "normal_term_true": "True",
        "normal_term_false": "False",
    },
    "ja": {
        "language_selector": "Language / 言語",
        "title": "Gaussian / SDF Dedupe App",
        "caption": f"Ver. {APP_VERSION}",
        "description": "Gaussian .log ファイルのSDF変換、および配座SDFファイルの読み込みを行い、RMSDに基づいて重複構造を除外します。",
        "info": (
            "Gaussian .log ファイルと、CONFLEX などが出力した配座 .sdf ファイルの両方に対応しています。"
            "SDF入力では、エネルギープロパティ名と単位を適切に設定してください。"
            "Gaussian と SDF を同時に投入する場合、エネルギー尺度が比較可能なときのみ順位づけに意味があります。"
        ),
        "settings": "設定",
        "developer_info": "開発者情報",
        "developer_name": "氏名",
        "developer_affiliation": "所属",
        "log_energy_header": "Gaussian .log の設定",
        "energy_to_extract": "Gaussian .log から抽出するエネルギー",
        "scf": "SCF",
        "free_energy": "自由エネルギー",
        "sdf_energy_header": "SDF の設定",
        "sdf_energy_mode": "SDF のエネルギープロパティ",
        "auto_detect": "代表的なエネルギープロパティを自動検出",
        "manual_property": "プロパティ名を手動指定",
        "sdf_property_name": "SDF のプロパティ名",
        "sdf_property_hint": "例: Energy, MMFF94 Energy, CONFLEX Energy",
        "sdf_energy_unit": "SDF のエネルギー単位",
        "hartree": "Hartree",
        "kcal_mol": "kcal/mol",
        "kj_mol": "kJ/mol",
        "unknown_unit": "不明（相対エネルギーを計算しない）",
        "symmetry_header": "RMSD の設定",
        "symmetry_handling": "対称性の扱い",
        "symmetric_present": "対称構造あり",
        "no_symmetric_concern": "対称構造を特に考慮しない",
        "rmsd_getbestrms": "RMSD法: GetBestRMS（対称性考慮）",
        "rmsd_alignmol": "RMSD法: AlignMol（直接アラインメント）",
        "rmsd_cutoff": "RMSD cutoff (Å)",
        "rmsd_cutoff_help": "この値未満のRMSDを示す配座は重複とみなします。",
        "heavy_atom_rmsd": "重原子RMSDを使用（水素を除去してRMSD計算）",
        "clear_results": "結果をクリア",
        "upload_files": "Gaussian .log および / または配座 .sdf ファイルをアップロード",
        "run_button": "処理とフィルタリングを実行",
        "need_file": ".log または .sdf ファイルを少なくとも1つアップロードしてください。",
        "obabel_not_available": "Open Babel が利用できません。",
        "processing": "処理中",
        "conversion_finished": "ファイル処理が完了しました。",
        "no_valid_molecules": "有効な構造を読み込めませんでした。",
        "results": "結果",
        "uploaded_files_metric": "アップロードファイル数",
        "loaded_conformers_metric": "読み込まれた配座数",
        "kept_structures_metric": "保持された構造数",
        "summary_table": "サマリーテーブル",
        "download_outputs": "出力ファイルのダウンロード",
        "download_all": "all_conformers.sdf をダウンロード",
        "download_unique": "unique_conformers.sdf をダウンロード",
        "download_summary": "summary.csv をダウンロード",
        "show_details": "処理詳細を表示",
        "no_details": "記録すべき処理上の問題はありませんでした。",
        "mixed_input_warning": (
            "Gaussian .log と SDF が同時に含まれています。"
            "すべてのエネルギーが比較可能な尺度にある場合にのみ、代表構造の選択に意味があります。"
        ),
        "status_kept": "保持",
        "status_removed_as_duplicate": "重複として除外",
        "status_energy_not_found": "エネルギー未検出",
        "status_conversion_failed": "変換失敗",
        "status_rdkit_read_failed": "RDKit 読み込み失敗",
        "status_unsupported_file_type": "非対応ファイル形式",
        "status_manual_property_not_found": "指定した SDF プロパティが見つからない",
        "normal_term_true": "True",
        "normal_term_false": "False",
    },
}

COLUMN_LABELS = {
    "en": {
        "source_label": "source_label",
        "source_file": "source_file",
        "input_type": "input_type",
        "record_index": "record_index",
        "energy_type": "energy_type",
        "energy_property": "energy_property",
        "energy_unit": "energy_unit",
        "energy_value": "energy_value",
        "relative_energy_kcal_mol": "relative_energy_kcal_mol",
        "rank_by_energy": "rank_by_energy",
        "status": "status",
        "duplicate_of": "duplicate_of",
        "rmsd_to_representative": "rmsd_to_representative",
        "normal_termination": "normal_termination",
    },
    "ja": {
        "source_label": "ソースラベル",
        "source_file": "元ファイル",
        "input_type": "入力形式",
        "record_index": "レコード番号",
        "energy_type": "エネルギー種別",
        "energy_property": "エネルギープロパティ",
        "energy_unit": "エネルギー単位",
        "energy_value": "エネルギー値",
        "relative_energy_kcal_mol": "相対エネルギー (kcal/mol)",
        "rank_by_energy": "エネルギー順位",
        "status": "状態",
        "duplicate_of": "重複先",
        "rmsd_to_representative": "代表構造へのRMSD",
        "normal_termination": "Gaussian正常終了",
    },
}

# =========================
# Session state initialization
# =========================
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

if "result_payload" not in st.session_state:
    st.session_state.result_payload = None

if "ui_language" not in st.session_state:
    st.session_state.ui_language = "English"


# =========================
# Helper functions
# =========================
def get_texts():
    lang = "ja" if st.session_state.ui_language == "日本語" else "en"
    return lang, UI_TEXT[lang]


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


def parse_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    match = FLOAT_RE.search(text)
    if not match:
        return None

    try:
        return float(match.group(0))
    except Exception:
        return None


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


def read_all_mols_from_sdf(sdf_path: Path):
    mols = []
    try:
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        for mol in suppl:
            if mol is not None:
                mols.append(mol)
    except Exception:
        return []
    return mols


def find_prop_name_case_insensitive(prop_dict, target_name):
    target_lower = target_name.strip().lower()
    for name in prop_dict:
        if name.lower() == target_lower:
            return name
    return None


def collect_numeric_props(mol):
    numeric_props = {}
    for name in mol.GetPropNames():
        value = parse_float(mol.GetProp(name))
        if value is not None:
            numeric_props[name] = value
    return numeric_props


def extract_energy_from_sdf_mol(mol, mode="auto", manual_property_name=""):
    numeric_props = collect_numeric_props(mol)

    if not numeric_props:
        return None, None, "energy_not_found"

    if mode == "manual":
        if not manual_property_name.strip():
            return None, None, "manual_property_not_found"

        matched_name = find_prop_name_case_insensitive(numeric_props, manual_property_name)
        if matched_name is None:
            return None, None, "manual_property_not_found"
        return numeric_props[matched_name], matched_name, "ok"

    for candidate in COMMON_SDF_ENERGY_PROPS:
        matched_name = find_prop_name_case_insensitive(numeric_props, candidate)
        if matched_name is not None:
            return numeric_props[matched_name], matched_name, "ok"

    keyword_priority = [
        "free",
        "gibbs",
        "energy",
        "scf",
        "mmff",
        "conflex",
        "forcefield",
        "enthalpy",
    ]

    scored = []
    for name, value in numeric_props.items():
        lname = name.lower()
        score = sum(1 for kw in keyword_priority if kw in lname)
        if score > 0:
            scored.append((score, name, value))

    if scored:
        scored.sort(key=lambda x: (-x[0], x[1].lower()))
        _, best_name, best_value = scored[0]
        return best_value, best_name, "ok"

    if len(numeric_props) == 1:
        only_name = next(iter(numeric_props))
        return numeric_props[only_name], only_name, "ok"

    return None, None, "energy_not_found"


def annotate_mol(
    mol,
    source_file,
    source_label,
    input_type,
    record_index,
    energy_value,
    energy_type,
    energy_unit,
    energy_source,
    normal_termination
):
    mol.SetProp(APP_SOURCE_FILE_PROP, str(source_file))
    mol.SetProp(APP_SOURCE_LABEL_PROP, str(source_label))
    mol.SetProp(APP_INPUT_TYPE_PROP, str(input_type))
    mol.SetProp(APP_RECORD_INDEX_PROP, str(record_index))
    mol.SetProp(APP_ENERGY_TYPE_PROP, str(energy_type))
    mol.SetProp(APP_ENERGY_UNIT_PROP, str(energy_unit))
    mol.SetProp(APP_ENERGY_SOURCE_PROP, str(energy_source))
    mol.SetProp(APP_NORMAL_TERM_PROP, str(normal_termination))

    if energy_value is not None:
        mol.SetProp(APP_ENERGY_PROP, str(energy_value))


def get_energy(mol):
    if mol.HasProp(APP_ENERGY_PROP):
        return parse_float(mol.GetProp(APP_ENERGY_PROP))
    return None


def get_prop_or_blank(mol, prop_name):
    return mol.GetProp(prop_name) if mol.HasProp(prop_name) else ""


def write_sdf_bytes(mols):
    sio = io.StringIO()
    writer = Chem.SDWriter(sio)
    for mol in mols:
        writer.write(mol)
    writer.close()
    return sio.getvalue().encode("utf-8")


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


def convert_relative_energy_to_kcal(delta_value, unit):
    unit = (unit or "").strip().lower()

    if unit == "hartree":
        return delta_value * ENERGY_HARTREE_TO_KCAL
    if unit == "kcal/mol":
        return delta_value
    if unit == "kj/mol":
        return delta_value * ENERGY_KJ_TO_KCAL

    return None


def deduplicate_molecules(
    mols,
    rmsd_cutoff=0.20,
    remove_hs_for_rmsd=True,
    rmsd_method="GetBestRMS"
):
    prepared = []
    for idx, mol in enumerate(mols, start=1):
        energy = get_energy(mol)
        source_file = get_prop_or_blank(mol, APP_SOURCE_FILE_PROP)
        source_label = get_prop_or_blank(mol, APP_SOURCE_LABEL_PROP) or f"mol_{idx}"
        input_type = get_prop_or_blank(mol, APP_INPUT_TYPE_PROP)
        record_index = get_prop_or_blank(mol, APP_RECORD_INDEX_PROP)
        energy_type = get_prop_or_blank(mol, APP_ENERGY_TYPE_PROP)
        energy_unit = get_prop_or_blank(mol, APP_ENERGY_UNIT_PROP)
        energy_source = get_prop_or_blank(mol, APP_ENERGY_SOURCE_PROP)
        normal_term = get_prop_or_blank(mol, APP_NORMAL_TERM_PROP)

        prepared.append({
            "mol": mol,
            "energy": energy,
            "source_file": source_file,
            "source_label": source_label,
            "input_type": input_type,
            "record_index": record_index,
            "energy_type": energy_type,
            "energy_unit": energy_unit,
            "energy_source": energy_source,
            "normal_termination": normal_term,
        })

    valid = [x for x in prepared if x["energy"] is not None]
    invalid = [x for x in prepared if x["energy"] is None]

    valid.sort(key=lambda x: x["energy"])

    kept_mols = []
    kept_info = []
    summary_rows = []

    for rank, item in enumerate(valid, start=1):
        mol = item["mol"]

        is_dup = False
        duplicate_of = ""
        best_rmsd = None

        for kept_item in kept_info:
            rmsd = calculate_rmsd(
                mol,
                kept_item["mol"],
                method=rmsd_method,
                remove_hs_for_rmsd=remove_hs_for_rmsd
            )

            if rmsd is not None and rmsd < rmsd_cutoff:
                is_dup = True
                duplicate_of = kept_item["source_label"]
                best_rmsd = rmsd
                break

        if not is_dup:
            kept_mols.append(mol)
            kept_info.append(item)

        summary_rows.append({
            "source_label": item["source_label"],
            "source_file": item["source_file"],
            "input_type": item["input_type"],
            "record_index": item["record_index"],
            "energy_type": item["energy_type"],
            "energy_property": item["energy_source"],
            "energy_unit": item["energy_unit"],
            "energy_value": item["energy"],
            "relative_energy_kcal_mol": None,
            "rank_by_energy": rank,
            "status": "removed_as_duplicate" if is_dup else "kept",
            "duplicate_of": duplicate_of,
            "rmsd_to_representative": best_rmsd,
            "normal_termination": item["normal_termination"],
        })

    for item in invalid:
        summary_rows.append({
            "source_label": item["source_label"],
            "source_file": item["source_file"],
            "input_type": item["input_type"],
            "record_index": item["record_index"],
            "energy_type": item["energy_type"],
            "energy_property": item["energy_source"],
            "energy_unit": item["energy_unit"],
            "energy_value": None,
            "relative_energy_kcal_mol": None,
            "rank_by_energy": None,
            "status": "energy_not_found",
            "duplicate_of": "",
            "rmsd_to_representative": None,
            "normal_termination": item["normal_termination"],
        })

    valid_units = {row["energy_unit"] for row in summary_rows if row["energy_value"] is not None}
    if len(valid_units) == 1:
        unit = next(iter(valid_units))
        valid_energies = [row["energy_value"] for row in summary_rows if row["energy_value"] is not None]
        if valid_energies:
            e0 = min(valid_energies)
            for row in summary_rows:
                if row["energy_value"] is not None:
                    delta = row["energy_value"] - e0
                    row["relative_energy_kcal_mol"] = convert_relative_energy_to_kcal(delta, unit)

    return kept_mols, summary_rows


def make_summary_csv_bytes(summary_rows):
    df = pd.DataFrame(summary_rows)
    csv_text = df.to_csv(index=False)
    return df, csv_text.encode("utf-8")


def localize_status(value, texts):
    mapping = {
        "kept": texts["status_kept"],
        "removed_as_duplicate": texts["status_removed_as_duplicate"],
        "energy_not_found": texts["status_energy_not_found"],
        "conversion_failed": texts["status_conversion_failed"],
        "rdkit_read_failed": texts["status_rdkit_read_failed"],
        "unsupported_file_type": texts["status_unsupported_file_type"],
        "manual_property_not_found": texts["status_manual_property_not_found"],
    }
    return mapping.get(value, value)


def make_display_df(raw_df, lang, texts):
    if raw_df is None or raw_df.empty:
        return raw_df

    df = raw_df.copy()

    if "status" in df.columns:
        df["status"] = df["status"].map(lambda x: localize_status(x, texts))

    rename_map = COLUMN_LABELS[lang]
    df = df.rename(columns=rename_map)
    return df


def render_results(payload, lang, texts):
    st.subheader(texts["results"])

    if payload.get("mixed_input_types", False):
        st.warning(texts["mixed_input_warning"])

    col1, col2, col3 = st.columns(3)
    col1.metric(texts["uploaded_files_metric"], payload["uploaded_count"])
    col2.metric(texts["loaded_conformers_metric"], payload["loaded_count"])
    col3.metric(texts["kept_structures_metric"], payload["kept_count"])

    st.subheader(texts["summary_table"])
    st.dataframe(make_display_df(payload["summary_df"], lang, texts), use_container_width=True)

    st.subheader(texts["download_outputs"])
    d1, d2, d3 = st.columns(3)

    d1.download_button(
        label=texts["download_all"],
        data=payload["all_sdf_bytes"],
        file_name="all_conformers.sdf",
        mime="chemical/x-mdl-sdfile"
    )

    d2.download_button(
        label=texts["download_unique"],
        data=payload["unique_sdf_bytes"],
        file_name="unique_conformers.sdf",
        mime="chemical/x-mdl-sdfile"
    )

    d3.download_button(
        label=texts["download_summary"],
        data=payload["summary_csv_bytes"],
        file_name="summary.csv",
        mime="text/csv"
    )

    with st.expander(texts["show_details"]):
        details_df = payload.get("processing_logs_df")
        if details_df is not None and not details_df.empty:
            st.dataframe(make_display_df(details_df, lang, texts), use_container_width=True)
        else:
            st.write(texts["no_details"])


# =========================
# Sidebar: language selector
# =========================
with st.sidebar:
    selected_language = st.selectbox(
        "Language / 言語",
        options=["English", "日本語"],
        index=0 if st.session_state.ui_language == "English" else 1,
    )
    st.session_state.ui_language = selected_language

lang, texts = get_texts()

# =========================
# Main header
# =========================
st.title(texts["title"])
st.caption(texts["caption"])
st.write(texts["description"])
st.info(texts["info"])

# =========================
# Sidebar settings
# =========================
with st.sidebar:
    st.header(texts["settings"])

    with st.expander(texts["developer_info"], expanded=False):
        st.write(f"**{texts['developer_name']}**: {DEVELOPER_INFO['name']}")
        if lang == "ja":
            st.write(f"**{texts['developer_affiliation']}**: {DEVELOPER_INFO['affiliation_ja']}")
        else:
            st.write(f"**{texts['developer_affiliation']}**: {DEVELOPER_INFO['affiliation_en']}")

    st.subheader(texts["log_energy_header"])
    energy_type_ui = st.radio(
        texts["energy_to_extract"],
        options=[texts["scf"], texts["free_energy"]],
        index=0,
    )
    log_energy_type_label = "SCF" if energy_type_ui == texts["scf"] else "Free Energy"

    st.subheader(texts["sdf_energy_header"])
    sdf_energy_mode_ui = st.radio(
        texts["sdf_energy_mode"],
        options=[texts["auto_detect"], texts["manual_property"]],
        index=0,
    )
    sdf_energy_mode = "auto" if sdf_energy_mode_ui == texts["auto_detect"] else "manual"

    sdf_manual_property_name = ""
    if sdf_energy_mode == "manual":
        sdf_manual_property_name = st.text_input(
            texts["sdf_property_name"],
            value="Energy",
            help=texts["sdf_property_hint"],
        )

    sdf_energy_unit_ui = st.selectbox(
        texts["sdf_energy_unit"],
        options=[texts["kcal_mol"], texts["hartree"], texts["kj_mol"], texts["unknown_unit"]],
        index=0,
    )
    sdf_energy_unit_map = {
        texts["hartree"]: "Hartree",
        texts["kcal_mol"]: "kcal/mol",
        texts["kj_mol"]: "kJ/mol",
        texts["unknown_unit"]: "Unknown",
    }
    sdf_energy_unit = sdf_energy_unit_map[sdf_energy_unit_ui]

    st.subheader(texts["symmetry_header"])
    symmetry_mode = st.radio(
        texts["symmetry_handling"],
        options=[
            texts["symmetric_present"],
            texts["no_symmetric_concern"],
        ],
        index=0,
    )

    if symmetry_mode == texts["symmetric_present"]:
        rmsd_method = "GetBestRMS"
        st.caption(texts["rmsd_getbestrms"])
    else:
        rmsd_method = "AlignMol"
        st.caption(texts["rmsd_alignmol"])

    rmsd_cutoff = st.number_input(
        texts["rmsd_cutoff"],
        min_value=0.01,
        max_value=10.00,
        value=0.20,
        step=0.01,
        format="%.2f",
        help=texts["rmsd_cutoff_help"]
    )

    remove_hs_for_rmsd = st.checkbox(
        texts["heavy_atom_rmsd"],
        value=True
    )

    if st.button(texts["clear_results"]):
        st.session_state.results_ready = False
        st.session_state.result_payload = None
        st.rerun()

# =========================
# Main UI
# =========================
uploaded_files = st.file_uploader(
    texts["upload_files"],
    type=["log", "sdf"],
    accept_multiple_files=True
)

run_button = st.button(texts["run_button"], type="primary")

# =========================
# Run
# =========================
if run_button:
    if not uploaded_files:
        st.error(texts["need_file"])
        st.stop()

    needs_obabel = any(Path(f.name).suffix.lower() == ".log" for f in uploaded_files)
    if needs_obabel:
        obabel_ok, obabel_msg = check_obabel_available()
        if not obabel_ok:
            st.error(texts["obabel_not_available"])
            st.code(obabel_msg)
            st.stop()
        st.success(obabel_msg)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        loaded_mols = []
        processing_logs = []

        progress = st.progress(0)
        status_box = st.empty()

        total_files = len(uploaded_files)
        input_types_seen = set()

        for i, uploaded_file in enumerate(uploaded_files, start=1):
            status_box.write(f"{texts['processing']} {i}/{total_files}: `{uploaded_file.name}`")

            suffix = Path(uploaded_file.name).suffix.lower()
            file_path = tmpdir / uploaded_file.name
            save_uploaded_file(uploaded_file, file_path)

            if suffix == ".log":
                input_types_seen.add("log")

                normal_term = check_normal_termination(file_path)
                energy_value = extract_last_scf_energy(file_path) if log_energy_type_label == "SCF" else extract_gibbs_energy(file_path)

                sdf_path = tmpdir / f"{file_path.stem}.sdf"
                conv_success, conv_stdout, conv_stderr = convert_log_to_sdf(file_path, sdf_path)

                if not conv_success:
                    processing_logs.append({
                        "source_label": uploaded_file.name,
                        "source_file": uploaded_file.name,
                        "input_type": "log",
                        "record_index": 1,
                        "energy_type": log_energy_type_label,
                        "energy_property": "",
                        "energy_unit": "Hartree",
                        "energy_value": energy_value,
                        "relative_energy_kcal_mol": None,
                        "rank_by_energy": None,
                        "status": "conversion_failed",
                        "duplicate_of": "",
                        "rmsd_to_representative": None,
                        "normal_termination": normal_term,
                    })
                    progress.progress(i / total_files)
                    continue

                mol = read_first_mol_from_sdf(sdf_path)
                if mol is None:
                    processing_logs.append({
                        "source_label": uploaded_file.name,
                        "source_file": uploaded_file.name,
                        "input_type": "log",
                        "record_index": 1,
                        "energy_type": log_energy_type_label,
                        "energy_property": "",
                        "energy_unit": "Hartree",
                        "energy_value": energy_value,
                        "relative_energy_kcal_mol": None,
                        "rank_by_energy": None,
                        "status": "rdkit_read_failed",
                        "duplicate_of": "",
                        "rmsd_to_representative": None,
                        "normal_termination": normal_term,
                    })
                    progress.progress(i / total_files)
                    continue

                annotate_mol(
                    mol=mol,
                    source_file=uploaded_file.name,
                    source_label=uploaded_file.name,
                    input_type="log",
                    record_index=1,
                    energy_value=energy_value,
                    energy_type=log_energy_type_label,
                    energy_unit="Hartree",
                    energy_source=log_energy_type_label,
                    normal_termination=normal_term,
                )
                loaded_mols.append(mol)

            elif suffix == ".sdf":
                input_types_seen.add("sdf")

                mols_in_sdf = read_all_mols_from_sdf(file_path)
                if not mols_in_sdf:
                    processing_logs.append({
                        "source_label": uploaded_file.name,
                        "source_file": uploaded_file.name,
                        "input_type": "sdf",
                        "record_index": "",
                        "energy_type": "SDF",
                        "energy_property": sdf_manual_property_name if sdf_energy_mode == "manual" else "",
                        "energy_unit": sdf_energy_unit,
                        "energy_value": None,
                        "relative_energy_kcal_mol": None,
                        "rank_by_energy": None,
                        "status": "rdkit_read_failed",
                        "duplicate_of": "",
                        "rmsd_to_representative": None,
                        "normal_termination": "",
                    })
                    progress.progress(i / total_files)
                    continue

                for idx, mol in enumerate(mols_in_sdf, start=1):
                    energy_value, prop_name, extract_status = extract_energy_from_sdf_mol(
                        mol,
                        mode=sdf_energy_mode,
                        manual_property_name=sdf_manual_property_name,
                    )

                    source_label = f"{uploaded_file.name} [record {idx}]"

                    annotate_mol(
                        mol=mol,
                        source_file=uploaded_file.name,
                        source_label=source_label,
                        input_type="sdf",
                        record_index=idx,
                        energy_value=energy_value,
                        energy_type="SDF",
                        energy_unit=sdf_energy_unit,
                        energy_source=prop_name or (sdf_manual_property_name if sdf_energy_mode == "manual" else ""),
                        normal_termination="",
                    )
                    loaded_mols.append(mol)

                    if extract_status == "manual_property_not_found":
                        processing_logs.append({
                            "source_label": source_label,
                            "source_file": uploaded_file.name,
                            "input_type": "sdf",
                            "record_index": idx,
                            "energy_type": "SDF",
                            "energy_property": sdf_manual_property_name,
                            "energy_unit": sdf_energy_unit,
                            "energy_value": None,
                            "relative_energy_kcal_mol": None,
                            "rank_by_energy": None,
                            "status": "manual_property_not_found",
                            "duplicate_of": "",
                            "rmsd_to_representative": None,
                            "normal_termination": "",
                        })

            else:
                processing_logs.append({
                    "source_label": uploaded_file.name,
                    "source_file": uploaded_file.name,
                    "input_type": "",
                    "record_index": "",
                    "energy_type": "",
                    "energy_property": "",
                    "energy_unit": "",
                    "energy_value": None,
                    "relative_energy_kcal_mol": None,
                    "rank_by_energy": None,
                    "status": "unsupported_file_type",
                    "duplicate_of": "",
                    "rmsd_to_representative": None,
                    "normal_termination": "",
                })

            progress.progress(i / total_files)

        status_box.write(texts["conversion_finished"])

        if not loaded_mols:
            st.error(texts["no_valid_molecules"])
            if processing_logs:
                st.dataframe(make_display_df(pd.DataFrame(processing_logs), lang, texts), use_container_width=True)
            st.stop()

        all_sdf_bytes = write_sdf_bytes(loaded_mols)

        kept_mols, summary_rows = deduplicate_molecules(
            mols=loaded_mols,
            rmsd_cutoff=rmsd_cutoff,
            remove_hs_for_rmsd=remove_hs_for_rmsd,
            rmsd_method=rmsd_method,
        )

        unique_sdf_bytes = write_sdf_bytes(kept_mols)

        summary_rows.extend(processing_logs)

        summary_df, summary_csv_bytes = make_summary_csv_bytes(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(
                by=["status", "rank_by_energy", "source_label"],
                na_position="last"
            ).reset_index(drop=True)

        processing_logs_df = pd.DataFrame(processing_logs) if processing_logs else pd.DataFrame()

        st.session_state.result_payload = {
            "uploaded_count": len(uploaded_files),
            "loaded_count": len(loaded_mols),
            "kept_count": len(kept_mols),
            "summary_df": summary_df,
            "all_sdf_bytes": all_sdf_bytes,
            "unique_sdf_bytes": unique_sdf_bytes,
            "summary_csv_bytes": summary_csv_bytes,
            "processing_logs_df": processing_logs_df,
            "mixed_input_types": len(input_types_seen) > 1,
        }
        st.session_state.results_ready = True

# render from session state
if st.session_state.results_ready and st.session_state.result_payload is not None:
    render_results(st.session_state.result_payload, lang, texts)
