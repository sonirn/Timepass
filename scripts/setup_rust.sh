#!/bin/bash
set -e

echo "Installing Rust..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

echo "Installing dependencies..."
apt-get update
apt-get install -y build-essential pkg-config libssl-dev

echo "Building the Ethereum recovery tool..."
cargo build --release

echo "Setup complete! The Ethereum recovery tool is ready to use."
