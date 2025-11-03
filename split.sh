#!/bin/bash

REQ=requirements.txt

# --------------- Parse requirements.txt ---------------
# - pypi_0 suffix -> pip package (package==version)
# - normal conda lines with builds -> conda package (pkg=version)
# - skip obviously bad lines (pure hashes / single chars / numeric garbage)
awk '
  /^[[:space:]]*#/ { next }                       # skip comments
  /^[[:space:]]*$/ { next }                       # skip empty
  {
    line=$0
    # If contains "=pypi_0" -> pip
    if (line ~ /=pypi_0$/) {
      # pkg=ver=pypi_0 -> pkg==ver
      sub(/=pypi_0$/,"",line)
      n=split(line, a, "="); pkg=a[1]; ver=a[2];
      if (pkg && ver) print pkg "==" ver >> "pip-requirements.txt"
      else print $0 >> "skipped-lines.txt"
    }
    # If looks like pkg=ver=build -> convert to pkg=ver for conda
    else if (line ~ /^[^=]+=[0-9]+\.[0-9]+.*=/) {
      # keep first two fields
      n=split(line,a,"="); pkg=a[1]; ver=a[2];
      if (pkg && ver) print pkg "=" ver >> "conda-pkgs.txt"
      else print $0 >> "skipped-lines.txt"
    }
    # If pkg=ver (no build) -> conda
    else if (line ~ /^[^=]+=[0-9]+\.[0-9]+/) {
      print line >> "conda-pkgs.txt"
    }
    # Lines that look like single words with no versions -> treat as pip installs later
    else if (line ~ /^[a-zA-Z0-9_.+-]+$/) {
      print line >> "pip-requirements.txt"
    }
    else {
      print $0 >> "skipped-lines.txt"
    }
  }
' $REQ

echo "Produced files:"
wc -l conda-pkgs.txt pip-requirements.txt || true
