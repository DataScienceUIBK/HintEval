import re
from pathlib import Path

version_file = Path("version/_version.py")
readme_file = Path("README.md")

version = None
with version_file.open("r") as f:
    content = f.read()
    match = re.search(r'__version__ = "(.+)"', content)
    if match:
        version = match.group(1)

if version:
    with readme_file.open("r") as f:
        readme_content = f.read()
    
    # Modify or add the version text as needed
    new_content = re.sub(r"x\.y\.z", version, readme_content)
    
    with readme_file.open("w") as f:
        f.write(new_content)
