#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8088}"
BACKEND_URL="${BACKEND_URL:-http://host.docker.internal:8080}"
DEVICE_ID="${DEVICE_ID:-PL-CB-SIM-0001}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d%H%M%S)}"
OUT_DIR="${OUT_DIR:-/tmp/cerebellum-smoke-${RUN_ID}}"
SYNC_RECEIVER_PORT="${SYNC_RECEIVER_PORT:-18088}"
mkdir -p "${OUT_DIR}"

pass=0
fail=0

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

json_payload() {
  local file="$1"
  jq -c . >"${OUT_DIR}/${file}.json"
}

request() {
  local name="$1"
  local method="$2"
  local path="$3"
  local expected="$4"
  local payload="${5:-}"
  local body="${OUT_DIR}/${name}.body"
  local code
  if [[ -n "${payload}" ]]; then
    code="$(curl -sS -o "${body}" -w '%{http_code}' -X "${method}" "${BASE_URL}${path}" \
      -H 'Content-Type: application/json' \
      --data-binary "@${payload}")"
  else
    code="$(curl -sS -o "${body}" -w '%{http_code}' -X "${method}" "${BASE_URL}${path}")"
  fi
  if [[ "${code}" == "${expected}" ]]; then
    log "PASS ${name} HTTP ${code}"
    pass=$((pass + 1))
  else
    log "FAIL ${name} HTTP ${code}, expected ${expected}"
    sed -n '1,20p' "${body}" || true
    fail=$((fail + 1))
  fi
}

request_check() {
  local name="$1"
  local method="$2"
  local path="$3"
  local expected="$4"
  local jq_expr="$5"
  local payload="${6:-}"
  request "${name}" "${method}" "${path}" "${expected}" "${payload}"
  local body="${OUT_DIR}/${name}.body"
  if jq -e "${jq_expr}" "${body}" >/dev/null; then
    log "PASS ${name} jq ${jq_expr}"
    pass=$((pass + 1))
  else
    log "FAIL ${name} jq ${jq_expr}"
    sed -n '1,20p' "${body}" || true
    fail=$((fail + 1))
  fi
}

log "Preparing local writable audio sample in cerebellum container"
docker exec patrol-cerebellum sh -lc \
  'ffmpeg -hide_banner -loglevel error -y -f lavfi -i anullsrc=channel_layout=mono:sample_rate=16000 -t 1 /var/lib/cerebellum/smoke-audio.wav'

log "Starting local HTTP sync receiver on port ${SYNC_RECEIVER_PORT}"
python3 -c '
import http.server
import sys

out_path = sys.argv[1]
port = int(sys.argv[2])

class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers.get("content-length", "0"))
        payload = self.rfile.read(size)
        with open(out_path, "ab") as output:
            output.write(payload + b"\n")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *_):
        return

http.server.HTTPServer(("0.0.0.0", port), Handler).serve_forever()
' "${OUT_DIR}/sync_receiver.jsonl" "${SYNC_RECEIVER_PORT}" &
sync_receiver_pid="$!"
trap 'kill "${sync_receiver_pid}" >/dev/null 2>&1 || true' EXIT
sleep 1

request_check health GET /health 200 '.status == "ok"'
request_check device_status GET /api/v1/device/status 200 '.device_id and .security.readonly_rootfs == true'
request_check events_initial GET /api/v1/events 200 '.count >= 0 and (.events | type == "array")'
request_check audit_initial GET /api/v1/audit 200 '.count >= 0 and (.records | type == "array")'
request_check cert_status GET /api/v1/security/certificates 200 'has("mtls_ready")'

jq -n --arg run "${RUN_ID}" '{source:"smoke-test", media_type:"image", duration_seconds:1, note:$run}' | json_payload media_ingest
request_check media_ingest POST /api/v1/media/ingest 200 '.accepted == true and .event.event_type == "media_ingest"' "${OUT_DIR}/media_ingest.json"

jq -n '{text:"帮我分析这段执法记录仪视频并生成摘要", media_type:"video", context:{mission_id:"smoke"}, candidate_functions:["video_summary","face_analyze","plate_analyze"]}' | json_payload function_recognize
request_check function_recognize POST /api/v1/functions/recognize 200 '.result.endpoint and .event.event_type == "function_recognized"' "${OUT_DIR}/function_recognize.json"

jq -n --arg run "${RUN_ID}" '{frame_id:("plate-" + $run), camera_id:"bodycam-smoke", image_uri:"/var/lib/cerebellum/samples/blank.ppm"}' | json_payload plate
request_check plate_analyze POST /api/v1/analyze/plate 200 '.result.frame_id and (.result.candidates | type == "array")' "${OUT_DIR}/plate.json"

jq -n --arg run "${RUN_ID}" '{frame_id:("face-" + $run), camera_id:"bodycam-smoke", image_uri:"/var/lib/cerebellum/samples/blank.ppm"}' | json_payload face
request_check face_analyze POST /api/v1/analyze/face 200 '.result.frame_id and (.result.faces | type == "array")' "${OUT_DIR}/face.json"

jq -n --arg run "${RUN_ID}" '{frame_id:("object-" + $run), camera_id:"bodycam-smoke", image_uri:"/var/lib/cerebellum/samples/blank.ppm", confidence_threshold:0.2, target_classes:["person","car"]}' | json_payload object
request_check object_analyze POST /api/v1/analyze/object 200 '.result.frame_id and (.result.detections | type == "array")' "${OUT_DIR}/object.json"

jq -n --arg run "${RUN_ID}" '{audio_uri:"/var/lib/cerebellum/smoke-audio.wav", mission_id:("mission-" + $run), language:"zh", max_tokens:100}' | json_payload asr
request_check asr_transcribe POST /api/v1/asr/transcribe 200 '.transcript.backend and .event.event_type == "audio_transcribed"' "${OUT_DIR}/asr.json"

jq -n --arg run "${RUN_ID}" '{file_uri:"/var/lib/cerebellum/samples/patrol-test.mp4", evidence_type:"video", mission_id:("mission-" + $run), encrypt:false, note:"smoke evidence"}' | json_payload evidence
request_check evidence_register POST /api/v1/evidence 200 '.evidence.evidence_id and .event.event_type == "evidence_registered"' "${OUT_DIR}/evidence.json"
request_check evidence_list GET /api/v1/evidence 200 '.count >= 1 and (.items | type == "array")'

jq -n --arg run "${RUN_ID}" --arg dest "http://host.docker.internal:${SYNC_RECEIVER_PORT}/sync" '{mission_id:("mission-" + $run), destination_url:$dest, include_events:true, include_audit:true, event_limit:20}' | json_payload sync_create
request_check sync_create POST /api/v1/sync/tasks 200 '.task.task_id and .event.event_type == "sync_task_created"' "${OUT_DIR}/sync_create.json"
sync_task_id="$(jq -r '.task.task_id' "${OUT_DIR}/sync_create.body")"
request_check sync_list GET /api/v1/sync/tasks 200 '.count >= 1 and (.tasks | type == "array")'
request_check sync_run POST "/api/v1/sync/tasks/${sync_task_id}/run" 200 '.task.status == "synced" and .event.event_type == "sync_task_run"'

embedding="$(jq -nc '[range(0;128) | if . == 0 then 1 else 0 end]')"
jq -n --arg run "${RUN_ID}" --argjson embedding "${embedding}" '{
  version:("smoke-face-lib-" + $run),
  source:"smoke-test",
  full_snapshot:false,
  model:"opencv-zoo-yunet+sface",
  persons:[{person_id:("SMOKE-" + $run), display_name:"Smoke Test", status:"ENABLED", risk_level:"LOW", category:"测试", embedding:$embedding}]
}' | json_payload face_apply
request_check face_library_apply POST /api/v1/face/library/apply 200 '.result.applied >= 1 and .event.event_type == "face_library_synced"' "${OUT_DIR}/face_apply.json"
request_check face_library_status GET /api/v1/face/library/status 200 'has("person_count") and has("pending_count")'

jq -n --arg backend "${BACKEND_URL}" --arg device "${DEVICE_ID}" '{backend_url:$backend, device_id:$device, force:true}' | json_payload face_sync
request_check face_library_sync POST /api/v1/face/library/sync 200 '.result.version and .event.event_type == "face_library_synced"' "${OUT_DIR}/face_sync.json"

stream_id="smoke-${RUN_ID}"
jq -n --arg stream "${stream_id}" '{
  stream_id:$stream,
  source_uri:"/var/lib/cerebellum/samples/patrol-test.mp4",
  camera_id:"bodycam-smoke",
  sample_fps:1,
  analyze_plate:true,
  analyze_face:true,
  analyze_object:true,
  max_runtime_seconds:5,
  max_analyzed_frames:2,
  save_sampled_frames:false
}' | json_payload stream_create
request_check stream_create POST /api/v1/streams 200 '.accepted == true and .stream.stream_id' "${OUT_DIR}/stream_create.json"
sleep 3
request_check stream_list GET /api/v1/streams 200 '.count >= 1 and (.streams | type == "array")'
request_check stream_get GET "/api/v1/streams/${stream_id}" 200 '.stream.stream_id'
request_check stream_stop POST "/api/v1/streams/${stream_id}/stop" 200 '.stream.status == "stopped" or .stream.status == "completed"'

jq -n --arg run "${RUN_ID}" --arg stream "${stream_id}" '{mission_id:("mission-" + $run), stream_id:$stream, operator_note:"smoke video summary", event_limit:50, use_llm:false, max_tokens:100}' | json_payload video_summary
request_check video_summary POST /api/v1/video/summary 200 '.summary.backend and .event.event_type == "video_summary_generated"' "${OUT_DIR}/video_summary.json"

jq -n --arg run "${RUN_ID}" '{mission_id:("mission-" + $run), report_type:"daily", prefer_quality:false, operator_note:"smoke report", max_tokens:100}' | json_payload llm_report
request_check llm_report POST /api/v1/llm/report 200 '.report.model and .event.event_type == "report_generated"' "${OUT_DIR}/llm_report.json"

request_check events_final GET /api/v1/events 200 '.count >= 10 and (.events | type == "array")'
request_check audit_final GET /api/v1/audit 200 '.count >= 10 and (.records | type == "array")'

log "Smoke output directory: ${OUT_DIR}"
log "Summary: pass=${pass}, fail=${fail}"

if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi
