# FCC Zone-Axis Finder

This project provides a Python/Tkinter GUI for indexing FCC TEM diffraction
patterns, predicting reachable zone axes on a double-tilt holder, and exploring
sample rotation, tilt, pole-figure, crystal, reciprocal-lattice, and diffraction
simulations.

## 1. Install Anaconda or Miniconda

Install one of the following:

- Anaconda: https://www.anaconda.com/download
- Miniconda: https://docs.conda.io/projects/miniconda/

After installation, open a terminal. On Windows, use **Anaconda Prompt** or a
terminal where `conda` is available.

## 2. Clone or Download This Project

Using Git:

```bash
git clone https://github.com/hepeng1024/zone_axis_finder
cd find_zone_axis
```

Or download the project ZIP from GitHub, unzip it, and open a terminal inside
the extracted `find_zone_axis` folder.

## 3. Create the Environment

The required Python packages are listed in `environment.yml`.

```bash
conda env create -f environment.yml
```

This creates a conda environment named `find-zone-axis`.

If the environment already exists and you want to update it:

```bash
conda env update -f environment.yml --prune
```

## 4. Activate the Environment

```bash
conda activate find-zone-axis
```

## 5. Run the GUI

From inside the project folder, run:

```bash
python zone_axis_finder_gui.py
```

The GUI will open as **FCC Zone-Axis Finder**. Choose an experimental diffraction
image, enter the current holder alpha/beta angles, adjust the options as needed,
and click **Run Analysis**.

## Project Files

- `zone_axis_finder_gui.py`: main graphical interface.
- `zone_axis_finder.py`: core indexing, matching, plotting, and tilt calculations.
- `environment.yml`: conda environment definition.
- `assets/`: artwork used by the GUI.

## Notes

- Keep the `assets/` folder in the same folder as `zone_axis_finder_gui.py`.
- The default conda environment name is `find-zone-axis`.
- If Tkinter does not open correctly, make sure you created the environment from
  `environment.yml` and activated it before running the GUI.
