"""Workstation lifecycle commands via boto3.

Manages the Isaac Sim GPU workstation: start, stop, status, ip, password, connect.
Resolves the instance via the Workstation stack output or by tag. All via boto3 (testable).
"""

from __future__ import annotations

import time

import boto3
import click
from botocore.exceptions import ClientError

from pai import config, helpers, cfn


@click.group()
def workstation():
    """Manage the Isaac Sim GPU workstation (start, stop, status, etc.)."""
    pass


def register(cli: click.Group) -> None:
    """Register workstation commands."""
    cli.add_command(workstation)


def _resolve_instance_id(region: str) -> str:
    """Resolve the workstation instance ID from the stack output or by tag.

    Args:
        region: AWS region

    Returns:
        Instance ID

    Raises:
        click.ClickException if instance not found
    """
    env = config.get("environment", "dev")
    stack_name = f"PhysicalAi-{env}-Workstation"

    # Try stack output first
    outputs = cfn.stack_outputs(stack_name, region)
    instance_id = outputs.get("WorkstationInstanceId")
    if instance_id:
        return instance_id

    # Fallback: search by tag
    try:
        ec2 = boto3.client("ec2", region_name=region)
        resp = ec2.describe_instances(
            Filters=[
                {"Name": "tag:Name", "Values": [f"physical-ai-{env}-workstation"]},
                {"Name": "instance-state-name", "Values": ["running", "stopped", "stopping", "pending"]},
            ]
        )
        for reservation in resp.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                return instance["InstanceId"]
    except Exception as e:
        helpers.friendly_boto_error(e)

    raise click.ClickException(
        f"Workstation instance not found. Stack {stack_name} may not be deployed.\n"
        "Deploy it first: pai deploy workstation"
    )


def _get_instance_info(instance_id: str, region: str) -> dict:
    """Get instance state, type, and public IP.

    Args:
        instance_id: EC2 instance ID
        region: AWS region

    Returns:
        Dict with keys: state, instance_type, public_ip (may be None if stopped)
    """
    try:
        ec2 = boto3.client("ec2", region_name=region)
        resp = ec2.describe_instances(InstanceIds=[instance_id])
        instance = resp["Reservations"][0]["Instances"][0]
        return {
            "state": instance["State"]["Name"],
            "instance_type": instance["InstanceType"],
            "public_ip": instance.get("PublicIpAddress"),
        }
    except Exception as e:
        helpers.friendly_boto_error(e)
        return {}


# ============================================================================
# Workstation subcommands
# ============================================================================


@workstation.command(name="start")
def start():
    """Start the workstation instance and print the new public IP.

    Note: The IP changes on every start (not an Elastic IP). Fetch it after start.
    """
    region = config.resolve_region()
    instance_id = _resolve_instance_id(region)

    helpers.info(f"Starting instance {instance_id} in {region}...")

    try:
        ec2 = boto3.client("ec2", region_name=region)

        # Check current state
        info = _get_instance_info(instance_id, region)
        state = info.get("state")

        if state == "running":
            helpers.success("Instance is already running.")
            public_ip = info.get("public_ip")
            if public_ip:
                helpers.info(f"Public IP: {public_ip}")
            return

        if state in ("pending", "stopping"):
            helpers.warn(f"Instance is {state}. Wait for it to stabilize, then retry.")
            return

        # Start the instance
        ec2.start_instances(InstanceIds=[instance_id])
        helpers.info("Waiting for instance to reach 'running' state...")

        # Poll until running (with timeout)
        waiter = ec2.get_waiter("instance_running")
        waiter.wait(InstanceIds=[instance_id], WaiterConfig={"Delay": 5, "MaxAttempts": 60})

        # Fetch the new public IP
        info = _get_instance_info(instance_id, region)
        public_ip = info.get("public_ip")

        helpers.success("\nWorkstation started successfully.")
        if public_ip:
            helpers.heading("Connection Details")
            helpers.info(f"  Public IP:  {public_ip}")
            helpers.info(f"  DCV URL:    https://{public_ip}:8443")
            helpers.info("  Username:   ubuntu")
            helpers.info("  Password:   pai-lab1  (change it: pai workstation password)")
            helpers.info("\n  Accept the self-signed certificate warning in your browser.")
        else:
            helpers.warn("No public IP assigned (unexpected). Check the EC2 console.")

    except Exception as e:
        helpers.friendly_boto_error(e)


@workstation.command(name="stop")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def stop(yes: bool):
    """Stop the workstation instance (halts billing).

    Note: The public IP will be released. A new IP is assigned on start.
    """
    region = config.resolve_region()
    instance_id = _resolve_instance_id(region)

    # Check current state
    info = _get_instance_info(instance_id, region)
    state = info.get("state")

    if state == "stopped":
        helpers.success("Instance is already stopped.")
        return

    if state in ("stopping", "pending"):
        helpers.warn(f"Instance is {state}. Wait for it to stabilize.")
        return

    if not yes:
        if not click.confirm(f"Stop instance {instance_id}? (billing halts, IP released)"):
            helpers.info("Aborted.")
            return

    helpers.info(f"Stopping instance {instance_id}...")

    try:
        ec2 = boto3.client("ec2", region_name=region)
        ec2.stop_instances(InstanceIds=[instance_id])
        helpers.success("Stop initiated. Billing will halt once the instance reaches 'stopped' state.")
    except Exception as e:
        helpers.friendly_boto_error(e)


@workstation.command(name="status")
def status():
    """Show workstation instance state, type, and public IP."""
    region = config.resolve_region()
    instance_id = _resolve_instance_id(region)

    info = _get_instance_info(instance_id, region)
    state = info.get("state", "unknown")
    instance_type = info.get("instance_type", "unknown")
    public_ip = info.get("public_ip") or "<none (instance stopped or starting)>"

    helpers.heading("Workstation Status")
    helpers.info(f"  Instance ID:  {instance_id}")
    helpers.info(f"  Region:       {region}")
    helpers.info(f"  State:        {state}")
    helpers.info(f"  Type:         {instance_type}")
    helpers.info(f"  Public IP:    {public_ip}")

    if state == "running" and info.get("public_ip"):
        helpers.info(f"\n  DCV URL:      https://{public_ip}:8443")


@workstation.command(name="ip")
def ip():
    """Print the current public IP (empty if stopped).

    Use this to re-fetch the IP after every start (it changes).
    """
    region = config.resolve_region()
    instance_id = _resolve_instance_id(region)

    info = _get_instance_info(instance_id, region)
    public_ip = info.get("public_ip")

    if public_ip:
        click.echo(public_ip)
    else:
        helpers.warn("No public IP (instance stopped or not yet assigned).")


@workstation.command(name="password")
@click.option("--new-password", help="New password for ubuntu user (default: pai-lab1)")
def password(new_password: str | None):
    """Set the DCV login password via SSM (or print the command for manual execution).

    The default password is 'pai-lab1'. Change it on first use.
    """
    region = config.resolve_region()
    instance_id = _resolve_instance_id(region)

    password_value = new_password or "pai-lab1"

    # Build the SetPassword command from the stack output (if available)
    env = config.get("environment", "dev")
    stack_name = f"PhysicalAi-{env}-Workstation"
    outputs = cfn.stack_outputs(stack_name, region)
    set_password_cmd = outputs.get("SetPassword")

    if set_password_cmd:
        # The stack output is a template with YOUR_PASSWORD placeholder
        # Replace it with the actual password
        actual_cmd = set_password_cmd.replace("YOUR_PASSWORD", password_value)
        helpers.info("Stack output provides SetPassword command:")
        helpers.info(f"  {actual_cmd}")
        helpers.info("\nRun this command to set the password, or use --new-password flag.")
    else:
        # Fallback: construct the command manually
        helpers.info("Construct the SSM send-command to set password:")
        helpers.info(f'  aws ssm send-command --instance-ids {instance_id} --region {region} \\')
        helpers.info(f'    --document-name "AWS-RunShellScript" \\')
        helpers.info(f'    --parameters \'commands=["echo ubuntu:{password_value} | chpasswd"]\'')

    # Optionally execute it via boto3 (commented out for safety — user should confirm)
    # Uncomment if you want the CLI to execute it directly:
    # try:
    #     ssm = boto3.client("ssm", region_name=region)
    #     ssm.send_command(
    #         InstanceIds=[instance_id],
    #         DocumentName="AWS-RunShellScript",
    #         Parameters={"commands": [f"echo ubuntu:{password_value} | chpasswd"]},
    #     )
    #     helpers.success(f"Password set to '{password_value}'.")
    # except Exception as e:
    #     helpers.friendly_boto_error(e)


@workstation.command(name="connect")
def connect():
    """Print connection details: DCV URL and SSM start-session command."""
    region = config.resolve_region()
    instance_id = _resolve_instance_id(region)

    info = _get_instance_info(instance_id, region)
    state = info.get("state")
    public_ip = info.get("public_ip")

    if state != "running":
        helpers.warn(f"Instance is {state}, not running. Start it first: pai workstation start")
        return

    if not public_ip:
        helpers.warn("No public IP assigned yet. Wait a moment and retry.")
        return

    helpers.heading("Workstation Connection")
    helpers.info(f"  DCV URL:      https://{public_ip}:8443")
    helpers.info("  Username:     ubuntu")
    helpers.info("  Password:     pai-lab1  (change it: pai workstation password)")
    helpers.info("\n  Accept the self-signed certificate warning in your browser.")
    helpers.info("\n  Shell access via SSM (no SSH key needed):")
    helpers.info(f"    aws ssm start-session --target {instance_id} --region {region}")
