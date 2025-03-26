#!/bin/bash
set -e

# Source Rust environment
source $HOME/.cargo/env

# Run the Ethereum recovery tool
./target/release/eth-recovery \
  --wordlist "$1" \
  --addresses "$2" \
  --output "$3" \
  --checkpoint "$4" \
  --derivation-paths "$5" \
  --chunk-size 1000000 "$6" \
  --threads "$7" \
  --checkpoint-interval "$8" \
  ${9:+--verbose}
