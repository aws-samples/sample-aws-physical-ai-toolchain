---
inclusion: auto
---

# Cosmos Deployment Details

## Current State
Cosmos Transfer 2.5-2B is running on a Spot p5.48xlarge in us-east-2.

## Instance Details
- **Instance ID:** i-03b2deb6b1e8116dc
- **Region:** us-east-2 (Ohio)
- **Type:** p5.48xlarge (8× H100 80GB)
- **Driver:** NVIDIA 580.159.03
- **Container:** 802782083985.dkr.ecr.us-east-1.amazonaws.com/physical-ai/cosmos-transfer:latest
- **Port:** 8000
- **Security Group:** sg-056c501bbd5a64dcd (port 8000 open)

## Key Lessons Learned

1. **SageMaker endpoints won't work for Cosmos** — SM p4d instances have NVIDIA driver 470 but Cosmos needs 580+. Use EC2 instead.

2. **Docker GPU runtime must be configured as default** — After installing nvidia-container-toolkit, you must set `"default-runtime": "nvidia"` in `/etc/docker/daemon.json` and restart Docker. Without this, `--gpus all` fails with "Error 802: system not yet initialized".

3. **Correct daemon.json:**
```json
{"default-runtime": "nvidia", "runtimes": {"nvidia": {"args": [], "path": "nvidia-container-runtime"}}}
```

4. **Reboot required** after nvidia-driver-550-server install for kernel module to load.

5. **p4d/p5 on-demand has no capacity** — use Spot instances. us-east-2 had cheapest Spot price ($7.64/hr for p5).

6. **IAM role needs Secrets Manager access** if you want the instance to pull NGC/NIM keys itself. Current workaround: pass key via SSM command parameter.

7. **Cross-region ECR pull works** but is slower (~5 min for the Cosmos image from us-east-1 to us-east-2).

## To Check Health
```bash
aws ssm send-command --instance-ids i-03b2deb6b1e8116dc \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["curl -s http://localhost:8000/v1/health/ready; echo ---; docker logs cosmos 2>&1 | tail -5"]' \
  --region us-east-2
```

## To Terminate (save costs)
```bash
aws ec2 terminate-instances --instance-ids i-03b2deb6b1e8116dc --region us-east-2
```

## API Format (Transfer2Request)
```json
{
  "prompt": "industrial warehouse with fluorescent lighting and metal shelving",
  "video": "<base64-encoded MP4 or URL>",
  "edge": {"enabled": true},
  "guidance": 3,
  "num_steps": 35,
  "resolution": "480",
  "seed": 42
}
```
Note: Cosmos Transfer operates on VIDEO (MP4), not single images. 
Integration: render Isaac Lab sim as MP4 → send to Cosmos → get photorealistic MP4 back.

## Additional p5 Requirement: nvidia-fabricmanager
p5 instances use NVSwitch for multi-GPU — CUDA won't initialize without fabricmanager:
```bash
apt-get install -yq nvidia-fabricmanager-550
systemctl enable nvidia-fabricmanager
systemctl start nvidia-fabricmanager
```

## Docker Flags Required
```bash
docker run -d --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ...
```

## For Toolkit Customers
The `cosmos_setup.py` script attempts SageMaker endpoint first. If that fails (driver issue), customers should:
1. Request p5 Spot capacity
2. Use the `scripts/cosmos-userdata.sh` bootstrap script
3. Or wait for SageMaker to update GPU drivers in their hosting fleet
