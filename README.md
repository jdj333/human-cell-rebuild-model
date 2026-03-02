# human-cell-rebuild-model

This project is a live-updating procedural renderer for a stylized human cell, implemented in Python using Matplotlib and NumPy. It visualizes cell components in stages, with a status UI and interactive updates.

## Features
- Procedural rendering of cell layers: membrane, nucleus, ER, Golgi, mitochondria, vesicles/ribosomes
- Live status updates between rendering stages
- Interactive Matplotlib figure (works in Jupyter, Colab, and local Python)
- Customizable seed and pause duration

## Requirements
- Python 3.8+
- numpy
- matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
To run the renderer:
```bash
python build.py
```

## Output
- The script opens a Matplotlib window and visualizes the cell in stages.
- Status and progress are shown during rendering.

## Repository
See the code in `build.py` for details on the rendering logic and customization options.
