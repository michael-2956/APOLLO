#!/bin/bash

# run from APOLLO dir

curl https://elan.lean-lang.org/elan-init.sh -sSf > elan.sh
chmod +x elan.sh
./elan.sh -y
source $HOME/.elan/env
elan toolchain install stable
elan default stable

cd repl && lake build