#!/bin/bash
# one resumable step with compact output: s.sh <arm> [budget]
timeout 172 python3 -u drive.py step --out runs/$1 --budget ${2:-164} --key-file $HOME/mnt/runpod-key/key.txt | sed -E 's/COMPLETED //; s/wall [0-9.]+ //; s/ ready [0-9.]+//; s/ gpu NVIDIA GeForce RTX 4090//; s/ err $//; s/exec ([0-9.]+) /e\1 /; s/delay /d/; s/worker ([a-z0-9]{4})[a-z0-9]+/w\1/; s/boot ([a-f0-9]{4})[a-f0-9]+/b\1/; s/ age [0-9.]+//' | cut -c1-110
