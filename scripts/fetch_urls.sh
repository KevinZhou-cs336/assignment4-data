#!/usr/bin/env bash
# Fetch a list of URLs into WARC files with several wget processes in parallel.
#
# Usage:
#     bash scripts/fetch_urls.sh data/wiki/sampled_urls.txt data/wiki/warc 16
set -euo pipefail

urls=$1
out_dir=$2
jobs=${3:-16}

mkdir -p "$out_dir/shards"
# Split into round-robin shards so each wget gets a similar mix of slow/fast hosts
awk -v n="$jobs" -v dir="$out_dir/shards" '{ printf "%s\n", $0 > sprintf("%s/urls_%02d", dir, (NR - 1) % n) }' "$urls"

for shard in "$out_dir"/shards/urls_*; do
    name=$(basename "$shard")
    # --tries=1: dead links are common, don't retry; || true: wget exits non-zero on any failed URL
    wget --timeout=5 --tries=1 --no-verbose \
        -i "$shard" \
        --warc-file="$out_dir/$name" \
        -O /dev/null \
        -o "$out_dir/$name.log" || true &
done
wait
echo "done: $(ls "$out_dir"/*.warc.gz | wc -l) WARC files in $out_dir"
