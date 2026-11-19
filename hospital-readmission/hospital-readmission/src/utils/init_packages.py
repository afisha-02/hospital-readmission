"""Create __init__.py files for all packages."""
from pathlib import Path

packages = [
    'src', 'src/data', 'src/validation', 'src/features', 'src/preprocessing',
    'src/models', 'src/calibration', 'src/thresholding', 'src/explainability',
    'src/fairness', 'src/monitoring', 'src/evaluation', 'app', 'app/ui', 'tests'
]

for pkg in packages:
    init_file = Path(pkg) / '__init__.py'
    if not init_file.exists():
        init_file.write_text('"""Package initialization."""\n', encoding='utf-8')
print("Initialized all packages.")
