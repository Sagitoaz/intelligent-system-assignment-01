"""Report Git candidate sizes without reading ignored local datasets."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
output = subprocess.check_output(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
    cwd=root,
)
paths = [root / item.decode("utf-8", errors="surrogateescape") for item in output.split(b"\0") if item]
files = sorted(((path.stat().st_size, path) for path in paths if path.is_file()), reverse=True)
for size, path in files[:15]:
    print(f"{size / 1024 / 1024:9.2f} MiB  {path.relative_to(root)}")
over_50 = [(size, path) for size, path in files if size > 50 * 1024 * 1024]
print("CANDIDATES_OVER_50_MIB", len(over_50))
assert not any(size >= 100 * 1024 * 1024 for size, _ in files), "Git candidate exceeds 100 MiB"
