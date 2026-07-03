"""Restart Cosmos NIM with correct profile_id (H100 fp8 latency)."""
import boto3, time

REGION = "us-east-2"
INSTANCE_ID = "i-YOUR_INSTANCE_ID"
ssm = boto3.client("ssm", region_name=REGION)

# Profile ID for H100 fp8 latency, from docker logs:
# Profile: d2b989bc5014fba468782bcade12f9fad9b67900e85e9cc833bfa431f7c7899b
# tags: {'gpu': 'h100', 'llm_precision': 'fp8', 'profile': 'latency'}
LATENCY_PROFILE_ID = "d2b989bc5014fba468782bcade12f9fad9b67900e85e9cc833bfa431f7c7899b"

fix_cmd = (
    "docker stop cosmos 2>/dev/null || true; "
    "docker rm cosmos 2>/dev/null || true; "
    "NGC_KEY=$(aws secretsmanager get-secret-value --secret-id ngc-api-key --region us-east-2 --query SecretString --output text); "
    f"docker run -d --gpus all --name cosmos "
    "--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:65536 "
    "-p 127.0.0.1:8000:8000 "
    f"-e NGC_API_KEY=\"$NGC_KEY\" -e NIM_MODEL_PROFILE={LATENCY_PROFILE_ID} "
    "nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:1.0.0 && echo 'Container started'"
)

resp = ssm.send_command(
    InstanceIds=[INSTANCE_ID],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": [fix_cmd]},
    TimeoutSeconds=120,
)
print(f"Restart command: {resp['Command']['CommandId']}")
time.sleep(25)
out = ssm.get_command_invocation(CommandId=resp["Command"]["CommandId"], InstanceId=INSTANCE_ID)
print(f"Status: {out.get('Status')} RC: {out.get('ResponseCode')}")
print(out.get("StandardOutputContent", "")[:400])
