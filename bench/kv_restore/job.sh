# Runs on the pod as /w/in/job_sweep.sh. MODE=full (write, restart, restore) or restore (a second pod on the same volume).
MODE=__MODE__
cd /w && tar xzf /w/in/bundle.tgz -C /w || { echo 'FAILED unpack' > /w/out/ALL_DONE; exit 1; }
export HF_HOME=/w/hf
B="python3 /w/bench.py --out /w/out/kv.jsonl --state /workspace/replies.json"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > /w/out/env.txt 2>&1 || { echo 'FAILED no gpu' > /w/out/ALL_DONE; exit 1; }
python3 -c "import vllm; print('vllm', vllm.__version__)" >> /w/out/env.txt 2>&1
(df -h /workspace /w; mount | grep -E ' /workspace | /w '; free -g | head -2; nproc) >> /w/out/env.txt 2>&1
[ -d /workspace ] || { echo 'FAILED no /workspace volume' > /w/out/ALL_DONE; exit 1; }

disk() {  # $1 dir $2 tag: direct-IO throughput, bypassing the page cache where the filesystem allows it
  mkdir -p $1/_dd
  (echo "== $2 write"; dd if=/dev/zero of=$1/_dd/big bs=1M count=2048 oflag=direct 2>&1 | tail -1
   echo "== $2 read direct"; dd if=$1/_dd/big of=/dev/null bs=1M iflag=direct 2>&1 | tail -1) >> /w/out/disk.txt 2>&1
  rm -rf $1/_dd
}
kvsize() { echo "$2 $(date -u +%T) files=$(find $1 -type f 2>/dev/null | wc -l) bytes=$(du -sb $1 2>/dev/null | cut -f1)" >> /w/out/kvsize.txt; }
readall() {  # $1 dir $2 tag: time a cold-as-possible read of every block file
  (sync; echo 3 > /proc/sys/vm/drop_caches) 2>/dev/null && echo "$2 drop_caches ok" >> /w/out/disk.txt || echo "$2 drop_caches NOT permitted" >> /w/out/disk.txt
  t0=$(date +%s.%N); find $1 -type f -print0 | xargs -0 -P 16 -n 64 cat > /dev/null; t1=$(date +%s.%N)
  echo "$2 readall seconds=$(awk "BEGIN{print $t1 - $t0}") bytes=$(du -sb $1 | cut -f1)" >> /w/out/disk.txt
}
start() {  # $1 model $2 kv root $3 log name $4 max model len $5 offload block size in tokens (optional)
  mkdir -p $2
  BS=""; [ -n "$5" ] && BS="\"block_size\":$5,"
  CFG="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"spec_name\":\"TieringOffloadingSpec\",\"cpu_bytes_to_use\":12884901888,$BS\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$2\"}]}}"
  t0=$(date +%s)
  setsid vllm serve $1 --host 127.0.0.1 --port 8001 --max-model-len $4 --served-model-name $1 --gpu-memory-utilization 0.90 \
    --max-num-seqs 8 --trust-remote-code --enable-prompt-tokens-details --kv-transfer-config "$CFG" > /w/out/$3 2>&1 &
  echo $! > /w/vllm.pid
  for i in $(seq 1 240); do
    curl -s -m 5 -H "Authorization: Bearer $VLLM_API_KEY" http://127.0.0.1:8001/v1/models | grep -q '"id"' && { echo "start $3 ready in $(( $(date +%s) - t0 )) s"; return 0; }
    kill -0 $(cat /w/vllm.pid) 2>/dev/null || { echo "start $3 FAILED: server exited"; tail -5 /w/out/$3; return 1; }
    sleep 5
  done
  echo "start $3 FAILED: not ready in 20 min"; return 1
}
stop() {
  kill -TERM -- -$(cat /w/vllm.pid) 2>/dev/null
  for i in $(seq 1 45); do kill -0 $(cat /w/vllm.pid) 2>/dev/null || break; sleep 2; done
  kill -KILL -- -$(cat /w/vllm.pid) 2>/dev/null; sleep 8
  rm -f /dev/shm/vllm_offload_*  # a killed server leaves its 12 GB CPU-tier region behind
}
pair() {  # $1 model $2 root $3 tag $4 first plan $5 second plan $6 max model len $7 block size: write, restart, restore
  start $1 $2 vllm_$3_s1.log $6 $7 && { $B --model $1 --label $3-s1 --plan $4; kvsize $2 $3-after-s1; sleep 60; kvsize $2 $3-after-s1-60s; }
  stop; kvsize $2 $3-after-stop; readall $2 $3
  start $1 $2 vllm_$3_s2.log $6 $7 && $B --model $1 --label $3-s2 --plan $5
  stop; kvsize $2 $3-after-s2
  echo "pair $3 done $(date -u)"
}

if [ "$MODE" = blocks ]; then  # needs a 48 GB GPU: 8B with 30k and 40k contexts (the model limit is 40,960), default 16-token blocks vs 512
  rm -rf /workspace/kv /workspace/kv_b16 /workspace/kv_b512; disk /workspace net
  pair Qwen/Qwen3-8B /workspace/kv_b512 q8-b512 warm,A30,B40 warm,A30,B40,E30,F40 40960 512
  rm -rf /workspace/kv_b512
  pair Qwen/Qwen3-8B /workspace/kv_b16 q8-b16 warm,A30,B40 warm,A30,B40,E30,F40 40960
  rm -rf /workspace/kv_b16
elif [ "$MODE" != restore ]; then
  disk /w local; disk /workspace net
  [ "$MODE" = full ] && pair Qwen/Qwen3-8B /w/kv_local q8-local warm,A8,A8,A30,A30 warm,A30,A30,A8,C30,X:A30 32000
  pair Qwen/Qwen3-8B /workspace/kv q8-net warm,A8,A8,A30,A30 warm,A30,A30,A8,C30,X:A30 32000
  rm -rf /w/hf/hub/models--Qwen--Qwen3-8B
  pair Qwen/Qwen3.5-9B /workspace/kv h9-net warm,A16,A16 warm,A16,A16,C16 20000
else
  disk /workspace net; kvsize /workspace/kv start
  readall /workspace/kv pod2-first-read
  start Qwen/Qwen3-8B /workspace/kv vllm_q8-net_p2.log 32000 && $B --model Qwen/Qwen3-8B --label q8-net-pod2 --plan warm,A30,A30,A8,D30
  stop; rm -rf /w/hf/hub/models--Qwen--Qwen3-8B
  start Qwen/Qwen3.5-9B /workspace/kv vllm_h9-net_p2.log 20000 && $B --model Qwen/Qwen3.5-9B --label h9-net-pod2 --plan warm,A16,D16
  stop
fi
grep -h -i -E "offload|tier|kv_connector|mamba" /w/out/vllm_*.log | grep -v -E "^\s*$" | head -80 > /w/out/vllm_offload_lines.txt
echo OK > /w/out/ALL_DONE
