# Contributing

## Git Setup (Amazon Internal)

### Clone
```bash
git clone git@ssh.gitlab.aws.dev:devris/aws-physical-ai-toolchain.git
```

### Push Issues
SSH to `gitlab.aws.dev` on port 22 times out on Amazon VPN (Route53 profile blocks it). Use `ssh.gitlab.aws.dev` instead:

```bash
# Fix remote if git push times out
git remote set-url origin git@ssh.gitlab.aws.dev:devris/aws-physical-ai-toolchain.git

# Verify
ssh -T git@ssh.gitlab.aws.dev
# Should print: Welcome to GitLab, @devris!
```

### Prerequisites
- Connected to Amazon VPN
- `mwinit` completed (signs SSH cert)
- Note: DCV workstation access (port 8443) requires VPN **disconnected** — VPN blocks non-standard ports

## Development Setup

```bash
cd cdk && npm install        # CDK dependencies
pip install -r training/requirements.txt  # Python dependencies for training scripts
```

## Key Docs

| Doc | What |
|-----|------|
| [PLAN.md](PLAN.md) | Task status, architecture decisions, what to work on next |
| [TESTING.md](TESTING.md) | Known issues when testing from fresh clone |
| [docs/cosmos-deployment-guide.md](docs/cosmos-deployment-guide.md) | Full Cosmos deployment instructions + troubleshooting |
| [docs/glossary.md](docs/glossary.md) | Physical AI terminology reference |
| [workshop/README.md](workshop/README.md) | Workshop overview + lab sequence |

## AWS Account

- Account: `<YOUR_ACCOUNT_ID>` (derive at runtime; do not hardcode in code — see `tests/test_repo_hygiene.py`)
- Region: `us-west-2` (primary, where the toolchain is validated)
- Secrets in Secrets Manager (regional — keep them in the deploy region): `physical-ai/ngc-api-key`, `physical-ai/nim-api-key`
