---
inclusion: auto
---

# GitLab Push Configuration

## Problem
SSH to `gitlab.aws.dev` on port 22 times out even when connected to Amazon VPN (Cisco AnyConnect / Route53 profile). The VPN profile blocks outbound SSH on port 22 to gitlab.aws.dev IPs.

## Solution
Use `ssh.gitlab.aws.dev` instead of `gitlab.aws.dev` as the git remote host. This host is already configured in `~/.ssh/config` (added by mwinit) with the correct cert, identity file, and ProxyCommand.

### Remote URL format
```
git@ssh.gitlab.aws.dev:devris/aws-physical-ai-toolchain.git
```

### Fix if remote is wrong
```bash
git remote set-url origin git@ssh.gitlab.aws.dev:devris/aws-physical-ai-toolchain.git
```

### Verify connectivity
```bash
ssh -T git@ssh.gitlab.aws.dev
# Should print: Welcome to GitLab, @devris!
```

## Web URL (for sharing)
https://gitlab.aws.dev/devris/aws-physical-ai-toolchain

## Prerequisites
- Connected to Amazon VPN
- mwinit completed (signs SSH cert)
