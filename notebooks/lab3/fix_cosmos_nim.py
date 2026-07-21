"""One-shot fix: pull correct NIM image on running instance."""
import sys
import boto3

REGION = "us-east-2"
INSTANCE_ID = sys.argv[1] if len(sys.argv) > 1 else ""

if not INSTANCE_ID:
    print("Usage: python fix_cosmos_nim.py <instance-id>")
    sys.exit(1)

ssm = boto3.client("ssm", region_name=REGION)

fix_cmd = r"""
docker stop cosmos 2>/dev/null || true
docker rm cosmos 2>/dev/null || true
NGC_KEY=$(aws secretsmanager get-secret-value --secret-id ngc-api-key --region us-east-2 --query SecretString --output text)
echo "$NGC_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
NIM_IMAGE="nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:1.0.0"
echo "Pulling $NIM_IMAGE"
docker pull "$NIM_IMAGE"
docker run -d --gpus all --name cosmos \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:65536 \
    -p 127.0.0.1:8000:8000 \
    -e NGC_API_KEY="$NGC_KEY" -e NIM_MODEL_PROFILE=latency \
    "$NIM_IMAGE"
echo "DONE"
"""

resp = ssm.send_command(
    InstanceIds=[INSTANCE_ID],
    DocumentName="AWS-RunShellScript",
    Parameters={"commands": [fix_cmd]},
    TimeoutSeconds=1800,
)
print(f"Command: {resp['Command']['CommandId']}")
