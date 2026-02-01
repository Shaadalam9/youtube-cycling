# youtube-cycling

## Citation and usage of code
If you use this work for academic work please cite the following paper:

> 

The code is open-source and free to use. It is aimed for, but not limited to, academic research. We welcome forking of this repository, pull requests, and any contributions in the spirit of open science and open-source code. For inquiries about collaboration, you may contact Md Shadab Alam (md_shadab_alam@outlook.com) or Pavlo Bazilinskyy (pavlo.bazilinskyy@gmail.com).

## Getting started
[![Python Version](https://img.shields.io/badge/python-3.11.9-blue.svg)](https://www.python.org/downloads/release/python-3919/)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-green)](https://docs.astral.sh/uv/)

Tested with **Python 3.11.9** and the [`uv`](https://docs.astral.sh/uv/) package manager.
Follow these steps to set up the project.

**Step 1:** Install `uv`. `uv` is a fast Python package and environment manager. Install it using one of the following methods:

**macOS / Linux (bash/zsh):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Alternative (if you already have Python and pip):**
```bash
pip install uv
```

**Step 2:** Fix permissions (if needed):

Sometimes `uv` needs to create a folder under `~/.local/share/uv/python` (macOS/Linux) or `%LOCALAPPDATA%\uv\python` (Windows).
If this folder was created by another tool (e.g. `sudo`), you may see an error like:
```lua
error: failed to create directory ... Permission denied (os error 13)
```

To fix it, ensure you own the directory:

### macOS / Linux
```bash
mkdir -p ~/.local/share/uv
chown -R "$(id -un)":"$(id -gn)" ~/.local/share/uv
chmod -R u+rwX ~/.local/share/uv
```

### Windows
```powershell
# Create directory if it doesn't exist
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\uv"

# Ensure you (the current user) own it
# (usually not needed, but if permissions are broken)
icacls "$env:LOCALAPPDATA\uv" /grant "$($env:UserName):(OI)(CI)F"
```

**Step 3:** After installing, verify:
```bash
uv --version
```

**Step 4:** Clone the repository:
```command line
git clone https://github.com/Shaadalam9/youtube-cycling.git
cd youtube-cycling
```

**Step 5:** Ensure correct Python version. If you don’t already have Python 3.11.9 installed, let `uv` fetch it:
```command line
uv python install 3.11.9
```
The repo should contain a .python-version file so `uv` will automatically use this version.

**Step 6:** Create and sync the virtual environment. This will create **.venv** in the project folder and install dependencies exactly as locked in **uv.lock**:
```command line
uv sync --frozen
```

**Step 7:** Activate the virtual environment:

**macOS / Linux (bash/zsh):**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```bat
.\.venv\Scripts\activate.bat
```

**Step 8:** Ensure that dataset are present. Place required datasets (including **mapping.csv**) into the **data/** directory:


**Step 9:** Run the code:
```command line
python3 analysis.py
```

### Configuration of project
Configuration of the project needs to be defined in `config`. Please use the `default.config` file for the required structure of the file. If no custom config file is provided, `default.config` is used. The config file has the following parameters:
- **`mapping`**: CSV file that contains mapping data for the cities referenced in the data.
- **`data`**: List of directories containing data (CSV output from YOLO).
- **`videos`**: Directory containing the videos used to generate the data.
- **`always_analyse`**: Always conduct analysis even when pickle files are present (good for testing).
- **`min_confidence`**: Sets the confidence threshold parameter for YOLO.
- **`countries_analyse`**: List of countries to include in the analysis. If empty, all countries are analysed.
- **`vehicles_analyse`**: List of YOLO class IDs to include in the analysis (e.g., `[0]`). If empty, all configured classes are analysed.
- **`DOWNLOAD_AND_ANNOTATE`**: If `true`, downloads (if needed) and generates annotated videos for relevant segments.
- **`KEEP_TRIMMED_CLIP`**: If `true`, keeps the intermediate trimmed clip on disk; otherwise it is deleted after the annotated video is written.
- **`ANNOTATE_ALL_BICYCLISTS`**: If `true`, always draw IDs for all detected bicyclists (not only those involved in following episodes).
- **`ANNOTATE_PAIR_OVERLAY`**: If `true`, overlays active follower → leader pair text on the video.
- **`CROP_AROUND_FOLLOWING`**: If `true`, when cycle-following is detected, outputs cropped annotated clips around each follower → leader pair.
- **`CROP_PRE_SECONDS`**: Number of seconds to include before the first encounter of the follower/leader pair.
- **`CROP_POST_GONE_SECONDS`**: Number of seconds to include after both follower and leader have disappeared from the frame.
- **`ALSO_WRITE_FULL_SEGMENT_WHEN_CROPPING`**: If `true`, also writes the full-segment annotated video in addition to the cropped clips.
- **`ANNOTATE_WHOLE_SEGMENT`**: If `true`, generates an annotated video for the **entire** CSV segment (download → trim to segment → annotate), regardless of whether cycle-following is detected. If used together with `CROP_AROUND_FOLLOWING=true`, the pipeline can output both full-segment annotated videos (for all segments) and cropped following clips (only for segments where following is detected).
- **`logger_level`**: Level of console output. Can be: debug, info, warning, error.
- **`font_family`**: Specifies the font family to be used in outputs.
- **`font_size`**: Specifies the font size to be used in outputs.
- **`plotly_template`**: Plotly template to use for figures (e.g., `plotly_white`).
- **`ftp_server`**: Base URL of the remote file server used to retrieve videos for annotation.