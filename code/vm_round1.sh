#!/bin/bash
# Narva Tester VM, round 1 (spec DOM-01-phaseB-v2.1): baseline, fullcontext (phase D), mem0, graphiti under one run id.
# Mem0's Postgres data (/var/narva-pg) and FalkorDB data (/var/narva-falkor) live on the VM disk; a rerun resumes.
exec > >(tee -a /var/log/narva-startup.log | logger -t narva) 2>&1
set -x
H=$(sha1sum "$0" | cut -c1-40)
gcloud logging write narva-vm-debug "start script=$0 H=$H done=$(cat /var/narva-done 2>/dev/null) ls=$(ls -la $0 2>&1)" --severity=INFO
[ "$(cat /var/narva-done 2>/dev/null)" = "$H" ] && { shutdown -h now; exit 0; }
IMAGE=us-central1-docker.pkg.dev/narva-rnd-engine/cloud-run-source-deploy/narva-tester-locomo:pb16
SMOKE=__SMOKE__
RUN_ID=__RUN_ID__
FLEX_TRIES=__FLEX__
SYSTEMS=baseline,fullcontext,mem0,graphiti
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh
docker system prune -af; df -h /
mkdir -p /var/narva-pg && chown 1000:1000 /var/narva-pg
docker rm -f falkor 2>/dev/null
docker run -d --name falkor --network host -v /var/narva-falkor:/var/lib/falkordb/data falkordb/falkordb:latest
for i in $(seq 1 60); do docker exec falkor redis-cli ping 2>/dev/null | grep -q PONG && break; sleep 2; done
gcloud auth configure-docker us-central1-docker.pkg.dev -q
docker run --rm --network host --log-driver=gcplogs -v /var/narva-pg:/var/narva-pg \
  -e GOOGLE_CLOUD_PROJECT=narva-rnd-engine -e RUNS_BUCKET=narva-rnd-engine-runs \
  -e SPEC_URI=gs://narva-rnd-engine-archive/benches/DOM-01-phaseB-v2.1.json \
  -e EXPECTED_SPEC_SHA=fc83e666b10da9a1ffe26ce9e7ab8817bfba4e734525b03f47c3e85a8f6fa547 \
  -e SMOKE=$SMOKE -e RUN_ID=$RUN_ID -e SYSTEMS=$SYSTEMS -e MEM0_TAG=bbd8ef -e GRAPHITI_TAG=1886b0 -e PGDATA_DIR=/var/narva-pg/data \
  -e MEM0_TELEMETRY=False -e GRAPHITI_TELEMETRY_ENABLED=false -e EMBEDDING_DIM=768 -e FALKOR_HOST=127.0.0.1 \
  -e CONCURRENCY=64 -e FULLCONTEXT_CONCURRENCY=12 -e FLEX_TRIES=$FLEX_TRIES -e RETRIEVAL_CONCURRENCY=8 \
  $IMAGE python phaseB.py
RC=$?
echo "phase B exit code: $RC"
docker exec falkor redis-cli SAVE
gcloud storage cp /var/log/narva-startup.log gs://narva-rnd-engine-runs/vm-logs/round1-startup-$(date -u +%Y%m%dT%H%M%SZ)-rc$RC.log
# the marker stops a finished script from running again on the next boot; a failed run must not set it,
# or every later boot with the same script shuts down at once (seen 17 Nov 20:55 and 18 Nov 09:41)
[ "$RC" = 0 ] && echo "$H" > /var/narva-done
shutdown -h now
