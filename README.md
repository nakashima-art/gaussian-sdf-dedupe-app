# Gaussian SDF Dedupe App

This Streamlit app:

- uploads multiple Gaussian `.log` files
- converts `.log` to `.sdf` with Open Babel
- extracts SCF or Gibbs energies
- merges all conformers into one SDF
- removes duplicate conformers by RMSD
- outputs:
  - `all_conformers.sdf`
  - `unique_conformers.sdf`
  - `summary.csv`
