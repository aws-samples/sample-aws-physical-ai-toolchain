# Lab 3 — Cosmos Transfer: Photorealistic Scene Generation

Generate photorealistic training scenes from Isaac Lab sim footage.
Optional lab — Lab 4 works without it.

---

## Quick Start

1. Read `BACKGROUND.md`
2. Check prerequisites (p5 quota + NGC key)
3. Run `Lab3_Cosmos_World_Generation.ipynb`
4. **Terminate the instance immediately when done** (~$7-8/hr)

---

## Files

```
lab3/
├── README.md
├── BACKGROUND.md
└── Lab3_Cosmos_World_Generation.ipynb
```

Supporting scripts (outside lab3, shared with toolchain):
```
training/scripts/cosmos_setup.py      ← launch/status/generate/terminate
training/scripts/cosmos3_generate.py  ← Cosmos 3 text/video generation (not yet validated)
scripts/cosmos-userdata.sh            ← EC2 bootstrap (driver + NIM)
docs/cosmos-deployment-guide.md       ← full runbook with troubleshooting
```

---

## Prerequisites

| Requirement | How to get it |
|-------------|--------------|
| `p5.48xlarge` Spot quota | Service Quotas console → EC2 → "Running On-Demand P instances" |
| NGC API key | https://ngc.nvidia.com/setup/api-key → store in Secrets Manager as `physical-ai/ngc-api-key` |
| IAM instance profile | `physical-ai-dev-cosmos-profile` with ECR pull + Secrets Manager read |

---

## Current Status

| Component | Status |
|-----------|--------|
| EC2 Spot p5 launch | ✅ Validated |
| Bootstrap (driver + NIM) | ✅ Validated |
| Health endpoint | ✅ Validated — `{"status":"ready"}` |
| Cosmos Transfer inference | ⚠️ API confirmed working, full restyle times out at CP=1 |
| CP=8 (all 8 H100s) | 🔲 Not yet tested — should fix timeout |
| Cosmos 3 generation | 🔲 Written, not validated |

---

## Cost

- p5.48xlarge Spot: ~$7-8/hr
- Typical session: ~$4-6 (15 min bootstrap + 30 min inference)
- **Always terminate when done**

## Lab 3 is Optional

Lab 4 (RL refinement) works without Cosmos using Isaac Lab's built-in domain
randomization. Add Cosmos only when sim-to-real transfer fails on real hardware
due to visual domain gap.
