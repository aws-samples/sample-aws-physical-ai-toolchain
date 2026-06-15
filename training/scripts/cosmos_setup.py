"""
Cosmos NIM Setup and Scene Generation for Physical AI Toolchain

Deploys Cosmos Transfer 2.5 (2B) as a SageMaker real-time endpoint for
photorealistic scene generation. Takes Isaac Lab sim renders and produces
photorealistic variations for training.

Usage:
    # Deploy Cosmos endpoint (one time)
    python cosmos_setup.py deploy

    # Generate scenes from sim renders
    python cosmos_setup.py generate --input ./sim_renders/ --output ./cosmos_scenes/

    # Delete endpoint when done (saves cost)
    python cosmos_setup.py teardown

Architecture:
    Cosmos Transfer 2.5-2B runs on H100 GPU via SageMaker endpoint.
    - Takes: sim-rendered image + control signal (edge/depth/segmentation)
    - Returns: photorealistic version of the same scene
    - Used for: domain randomization enhancement (Lab 3)

Cost:
    - p4d.24xlarge endpoint: ~$32/hr while running
    - Generate 100 scenes: ~5 min = ~$2.70
    - Remember to teardown when done!

Why Cosmos (optional enhancement):
    Isaac Lab's built-in domain randomization (random positions, lighting, colors)
    achieves ~85-90% sim-to-real transfer for most manipulation tasks.
    
    Cosmos adds photorealistic visual diversity:
    - Realistic material textures (scratched metal, plastic, wood grain)
    - Complex lighting (mixed sources, shadows, reflections)
    - Environmental context (cluttered backgrounds, dust, wear)
    
    This pushes sim-to-real transfer to ~95%+ for visually challenging tasks.
    
    You DON'T need Cosmos if:
    - Your task is position-based (not appearance-sensitive)
    - Built-in randomization gives acceptable real-world performance
    - You're in early prototyping (get it working first, add Cosmos later)
    
    You DO need Cosmos if:
    - Policy fails on real hardware due to visual domain gap
    - Task involves color/texture discrimination (sorting, defect detection)
    - Environment has complex backgrounds that confuse the camera
"""

import argparse
import boto3
import json
import os
import sys
import time

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
ENDPOINT_NAME = "physical-ai-cosmos-transfer"
MODEL_NAME = "cosmos-transfer-2-5-2b"
INSTANCE_TYPE = "ml.p4d.24xlarge"  # 8x A100 80GB (Cosmos Transfer 2.5 needs H100 or A100)

# Cosmos NIM container from our ECR (pulled from NGC via CodeBuild)
COSMOS_IMAGE = "802782083985.dkr.ecr.us-east-1.amazonaws.com/physical-ai/cosmos-transfer:latest"


def get_ngc_key():
    """Retrieve NGC API key from Secrets Manager."""
    sm = boto3.client("secretsmanager", region_name=REGION)
    return sm.get_secret_value(SecretId="physical-ai/ngc-api-key")["SecretString"]


def get_nim_key():
    """Retrieve NIM API key from Secrets Manager."""
    sm = boto3.client("secretsmanager", region_name=REGION)
    return sm.get_secret_value(SecretId="physical-ai/nim-api-key")["SecretString"]


def deploy_endpoint():
    """Deploy Cosmos Transfer 2.5 as a SageMaker endpoint."""
    sagemaker = boto3.client("sagemaker", region_name=REGION)
    
    # Get the SageMaker execution role
    iam = boto3.client("iam", region_name=REGION)
    role_arn = f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/physical-ai-dev-sagemaker-role"
    
    ngc_key = get_ngc_key()
    
    print(f"{'='*60}")
    print(f"  Deploying Cosmos Transfer 2.5-2B")
    print(f"  Instance: {INSTANCE_TYPE}")
    print(f"  Endpoint: {ENDPOINT_NAME}")
    print(f"  Cost: ~$32/hr while running")
    print(f"{'='*60}")
    
    # Step 1: Create SageMaker Model
    # Using the NIM container pattern — NIM auto-downloads model weights on first start
    print("\n  Creating SageMaker model...")
    try:
        sagemaker.create_model(
            ModelName=MODEL_NAME,
            PrimaryContainer={
                "Image": COSMOS_IMAGE,
                "Environment": {
                    "NGC_API_KEY": ngc_key,
                    "NIM_MODEL_SIZE": "2b",
                },
            },
            ExecutionRoleArn=role_arn,
        )
    except sagemaker.exceptions.ClientError as e:
        if "Cannot create already existing model" in str(e):
            print("  Model already exists, skipping...")
        else:
            raise
    
    # Step 2: Create endpoint config
    print("  Creating endpoint configuration...")
    config_name = f"{ENDPOINT_NAME}-config"
    try:
        sagemaker.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                "VariantName": "primary",
                "ModelName": MODEL_NAME,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
                "ContainerStartupHealthCheckTimeoutInSeconds": 900,  # NIM takes time to load
            }],
        )
    except sagemaker.exceptions.ClientError as e:
        if "Cannot create already existing" in str(e):
            print("  Config already exists, skipping...")
        else:
            raise
    
    # Step 3: Create endpoint
    print("  Creating endpoint (this takes 10-15 min for model loading)...")
    try:
        sagemaker.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )
    except sagemaker.exceptions.ClientError as e:
        if "Cannot create already existing" in str(e):
            print("  Endpoint already exists.")
            return
        else:
            raise
    
    # Wait for endpoint
    print("  Waiting for endpoint to be InService...")
    waiter = sagemaker.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=ENDPOINT_NAME,
        WaiterConfig={"Delay": 30, "MaxAttempts": 60}
    )
    print("  ✅ Endpoint is live!")
    print(f"\n  Endpoint: {ENDPOINT_NAME}")
    print(f"  To stop: python cosmos_setup.py teardown")


def teardown_endpoint():
    """Delete the Cosmos endpoint to save cost."""
    sagemaker = boto3.client("sagemaker", region_name=REGION)
    
    print("  Deleting Cosmos endpoint...")
    try:
        sagemaker.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print("  Endpoint deleted.")
    except Exception as e:
        print(f"  {e}")
    
    try:
        sagemaker.delete_endpoint_config(EndpointConfigName=f"{ENDPOINT_NAME}-config")
        print("  Endpoint config deleted.")
    except Exception as e:
        print(f"  {e}")
    
    try:
        sagemaker.delete_model(ModelName=MODEL_NAME)
        print("  Model deleted.")
    except Exception as e:
        print(f"  {e}")
    
    print("  ✅ Cosmos resources cleaned up. No more charges.")


def generate_scenes(input_dir: str, output_dir: str, num_variations: int = 4):
    """
    Generate photorealistic scene variations from sim renders.
    
    Takes rendered images from Isaac Lab and uses Cosmos Transfer to
    produce photorealistic versions with different styles.
    """
    import base64
    from pathlib import Path
    
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Style prompts for domain randomization
    styles = [
        "industrial warehouse with fluorescent overhead lighting, metal shelving, concrete floor",
        "modern factory floor with natural skylights, white walls, robotic cells",
        "dimly lit workshop with task lighting, oil-stained surfaces, tool racks on walls",
        "clean room environment with bright even lighting, stainless steel surfaces",
        "aged manufacturing facility with worn paint, mixed lighting, cluttered background",
    ]
    
    image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
    
    if not image_files:
        print(f"  No images found in {input_dir}")
        print(f"  Generate sim renders first with Isaac Lab, then run this.")
        return
    
    print(f"  Found {len(image_files)} input images")
    print(f"  Generating {num_variations} variations each")
    print(f"  Output: {output_dir}")
    
    total = 0
    for img_file in image_files:
        with open(img_file, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        
        for i, style in enumerate(styles[:num_variations]):
            payload = json.dumps({
                "input_image": img_b64,
                "prompt": style,
                "control_type": "edge",  # Use edge detection as control signal
                "strength": 0.7,  # Balance between original structure and style
                "seed": i * 42,
            })
            
            try:
                response = runtime.invoke_endpoint(
                    EndpointName=ENDPOINT_NAME,
                    ContentType="application/json",
                    Body=payload,
                )
                
                result = json.loads(response["Body"].read())
                output_img = base64.b64decode(result["output_image"])
                
                out_name = f"{img_file.stem}_style{i:02d}.png"
                with open(output_path / out_name, "wb") as f:
                    f.write(output_img)
                
                total += 1
                print(f"    Generated: {out_name}")
                
            except Exception as e:
                print(f"    Error generating {img_file.name} style {i}: {e}")
    
    print(f"\n  ✅ Generated {total} photorealistic scene variations")
    print(f"  Output: {output_path}")
    print(f"\n  Next: Upload to S3 for RL training:")
    print(f"    aws s3 sync {output_path} s3://physical-ai-dev-datasets-802782083985/cosmos-scenes/")


def status():
    """Check Cosmos endpoint status."""
    sagemaker = boto3.client("sagemaker", region_name=REGION)
    try:
        resp = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"  Endpoint: {ENDPOINT_NAME}")
        print(f"  Status: {resp['EndpointStatus']}")
        print(f"  Instance: {INSTANCE_TYPE}")
        if resp["EndpointStatus"] == "InService":
            print(f"  ⚠️  Running — costing ~$32/hr. Run 'teardown' when done.")
    except sagemaker.exceptions.ClientError:
        print(f"  Endpoint '{ENDPOINT_NAME}' not found. Run 'deploy' first.")


def main():
    parser = argparse.ArgumentParser(description="Cosmos NIM Setup for Physical AI Toolchain")
    parser.add_argument("action", choices=["deploy", "teardown", "generate", "status"],
                        help="Action to perform")
    parser.add_argument("--input", default="./sim_renders/",
                        help="Input directory with sim-rendered images")
    parser.add_argument("--output", default="./cosmos_scenes/",
                        help="Output directory for generated scenes")
    parser.add_argument("--variations", type=int, default=4,
                        help="Number of style variations per image")
    args = parser.parse_args()
    
    if args.action == "deploy":
        deploy_endpoint()
    elif args.action == "teardown":
        teardown_endpoint()
    elif args.action == "generate":
        generate_scenes(args.input, args.output, args.variations)
    elif args.action == "status":
        status()


if __name__ == "__main__":
    main()
