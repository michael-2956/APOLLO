#!/bin/bash

# run from APOLLO dir

curl https://elan.lean-lang.org/elan-init.sh -sSf > elan.sh
chmod +x elan.sh
./elan.sh -y
source $HOME/.elan/env
elan toolchain install leanprover/lean4:v4.17.0
elan default leanprover/lean4:v4.17.0

cd repl && lake build