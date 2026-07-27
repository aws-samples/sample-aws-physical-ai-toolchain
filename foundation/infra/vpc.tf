# =============================================================================
# VPC: Shared networking for GPU Batch compute (Isaac Lab, GR00T, etc.)
# 2 public subnets (NAT gateway) + 3 private subnets (Batch instances)
# =============================================================================

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "main" {
  cidr_block           = "10.1.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name        = "${local.prefix}-vpc"
    Project     = var.project_name
    Environment = var.environment
  }
}

# --- Public Subnets ---

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name        = "${local.prefix}-public-${count.index + 1}"
    Project     = var.project_name
    Environment = var.environment
  }
}

# --- Private Subnets ---

resource "aws_subnet" "private" {
  count             = 3
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name        = "${local.prefix}-private-${count.index + 1}"
    Project     = var.project_name
    Environment = var.environment
  }
}

# --- Internet Gateway ---

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name        = "${local.prefix}-igw"
    Project     = var.project_name
    Environment = var.environment
  }
}

# --- Elastic IP for NAT Gateway ---

resource "aws_eip" "nat" {
  domain = "vpc"

  tags = {
    Name        = "${local.prefix}-nat-eip"
    Project     = var.project_name
    Environment = var.environment
  }
}

# --- NAT Gateway (single, in first public subnet) ---

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id

  tags = {
    Name        = "${local.prefix}-nat"
    Project     = var.project_name
    Environment = var.environment
  }

  depends_on = [aws_internet_gateway.main]
}

# --- Route Tables ---

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name        = "${local.prefix}-public-rt"
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }

  tags = {
    Name        = "${local.prefix}-private-rt"
    Project     = var.project_name
    Environment = var.environment
  }
}

# --- Route Table Associations ---

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = 3
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# --- SSM Parameters (so downstream components discover the VPC) ---

resource "aws_ssm_parameter" "vpc_id" {
  name  = "/${var.project_name}/vpc-id"
  type  = "String"
  value = aws_vpc.main.id
}

resource "aws_ssm_parameter" "private_subnet_ids" {
  name  = "/${var.project_name}/private-subnet-ids"
  type  = "StringList"
  value = join(",", aws_subnet.private[*].id)
}
