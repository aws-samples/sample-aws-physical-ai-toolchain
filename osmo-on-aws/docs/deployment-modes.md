# OSMO on AWS - Deployment Modes

OSMO supports three deployment modes to accommodate different architecture requirements.

## Overview

| Mode | Use Case | Components |
|------|----------|------------|
| Full | Single cluster with all components | Control Plane + Compute |
| Control-Plane-Only | Central management cluster | Control Plane only |
| Backend-Only | Remote compute clusters | Compute only |

## Full Mode (Default)

Deploys everything in a single EKS cluster.

### Infrastructure
- VPC with public/private subnets
- EKS cluster with system and GPU node groups
- RDS PostgreSQL
- ElastiCache Redis
- S3 buckets for workflows and datasets

### OSMO Components
- OSMO Service (API)
- OSMO Router
- OSMO Web UI
- OSMO Backend Operator

### Configuration

```hcl
# terraform.tfvars
deployment_mode = "full"
should_deploy_postgresql = true
should_deploy_redis = true
should_deploy_gpu_nodes = true
```

### Deployment

```bash
# Run all scripts
./01-deploy-aws-prerequisites.sh
./02-deploy-gpu-infrastructure.sh
./03-deploy-keycloak.sh --admin-password 'YourSecurePassword!'
./04-deploy-osmo-control-plane.sh --ngc-api-key "your-key"   # applies declarative ConfigMap config
./05-deploy-osmo-backend.sh                                  # creates the osmo-workflow IRSA SA
```

## Control-Plane-Only Mode

For centralized management with remote compute clusters.

### Infrastructure
- VPC with public/private subnets
- EKS cluster with system nodes only (no GPU nodes)
- RDS PostgreSQL
- ElastiCache Redis
- S3 buckets

### OSMO Components
- Consolidated `service` chart: OSMO Service (API) + Router + Web UI + gateway
- No Backend Operator (remote backends connect to this control plane)

### Configuration

```hcl
# terraform.tfvars
deployment_mode = "control-plane-only"
should_deploy_postgresql = true
should_deploy_redis = true
should_deploy_gpu_nodes = false
```

### Deployment

```bash
./01-deploy-aws-prerequisites.sh
# Skip GPU operator
./03-deploy-keycloak.sh --admin-password 'YourSecurePassword!'
./04-deploy-osmo-control-plane.sh --ngc-api-key "your-key"   # applies declarative ConfigMap config
# Skip backend operator
```

### Remote Backend Registration

After control plane deployment, obtain the service URL and use it when deploying backend-only clusters:

```bash
# Get control plane URL
kubectl get ingress -n osmo -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}'
```

## Backend-Only Mode

For distributed compute clusters that connect to a central control plane.

### Infrastructure
- VPC with public/private subnets
- EKS cluster with system and GPU node groups
- S3 buckets (for local workflow data caching)
- No RDS or ElastiCache (uses control plane's databases)

### OSMO Components
- OSMO Backend Operator
- No Control Plane components

### Configuration

```hcl
# terraform.tfvars
deployment_mode = "backend-only"
should_deploy_postgresql = false
should_deploy_redis = false
should_deploy_gpu_nodes = true
external_service_url = "https://osmo-control-plane.example.com"
```

### Deployment

```bash
./01-deploy-aws-prerequisites.sh
./02-deploy-gpu-infrastructure.sh
# Skip control plane
./05-deploy-osmo-backend.sh --service-url "https://osmo-control-plane.example.com"
```

### Backend Token

The backend operator needs a service token from the control plane:

```bash
# On control plane cluster
osmo token create backend-operator --expires-in 365d

# On backend cluster
kubectl create secret generic osmo-operator-token \
  -n osmo \
  --from-literal=token="<TOKEN_FROM_CONTROL_PLANE>"
```

## Multi-Region Architecture

For global deployments:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Control Plane (us-west-2)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ OSMO Service │  │ OSMO Router  │  │  OSMO Web UI │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│           │                                                      │
│  ┌────────┴────────┐  ┌──────────────┐                         │
│  │  RDS PostgreSQL │  │ ElastiCache  │                         │
│  └─────────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Backend (us-east-1)│  │ Backend (eu-west-1)│  │ Backend (ap-northeast-1)│
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │Backend Operator│ │  │ │Backend Operator│ │  │ │Backend Operator│ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │ GPU Nodes   │ │  │ │ GPU Nodes   │ │  │ │ GPU Nodes   │ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Considerations

### Network Connectivity

- Backend-only clusters need network connectivity to the control plane
- Consider VPC peering, Transit Gateway, or public endpoints
- WebSocket connections (wss://) are used for real-time communication

### Security

- Use private endpoints where possible
- Implement network policies to restrict traffic
- Rotate backend operator tokens regularly
- Use AWS PrivateLink for cross-region connectivity

### Latency

- Place backends close to data sources
- Consider S3 replication for frequently accessed datasets
- Monitor WebSocket connection latency

### Cost Optimization

- Scale GPU nodes to zero when not in use
- Use Spot instances for fault-tolerant workloads
- Share control plane across multiple backend clusters
