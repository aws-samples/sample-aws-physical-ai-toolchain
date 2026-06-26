"""Config show/set commands.

Read and write config.json. Mask sensitive values (allowedCidr) in show output.
"""

from __future__ import annotations

import json

import click

from pai import config, helpers


@click.group(name="config")
def config_group():
    """View and edit config.json settings."""
    pass


def register(cli: click.Group) -> None:
    """Register config commands."""
    cli.add_command(config_group)


@config_group.command(name="show")
def show():
    """Pretty-print the resolved config.json (with resolved region).

    GUARDRAIL: allowedCidr is masked as <redacted> (it's the user's personal IP).
    """
    cfg = config.load()
    region = config.resolve_region()

    # Mask allowedCidr in workstation section (if present)
    if "workstation" in cfg and "allowedCidr" in cfg["workstation"]:
        cfg["workstation"]["allowedCidr"] = "<redacted>"

    helpers.heading("Configuration")
    helpers.info(f"Resolved region: {region}")
    helpers.info(f"Config path:     {config.CONFIG_PATH}\n")

    # Pretty-print the config
    click.echo(json.dumps(cfg, indent=2))


@config_group.command(name="set")
@click.argument("key")
@click.argument("value")
def set_config(key: str, value: str):
    """Set a config value by dotted key (e.g., aws.region us-west-2).

    Preserves _comment keys and formatting as much as practical.
    """
    cfg = config.load()

    # Parse dotted key and navigate to the parent
    keys = key.split(".")
    parent = cfg
    for k in keys[:-1]:
        if k not in parent:
            parent[k] = {}
        parent = parent[k]

    # Set the final key
    final_key = keys[-1]
    old_value = parent.get(final_key)
    parent[final_key] = value

    # Write back to config.json
    config.save(cfg)

    helpers.success(f"Set {key} = {value}")
    if old_value:
        helpers.info(f"  (previous value: {old_value})")
