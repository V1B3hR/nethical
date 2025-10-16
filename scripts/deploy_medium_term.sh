#!/bin/bash
# Deploy Nethical to all 10 regions for medium-term scalability
# 
# This script deploys Nethical governance system to 10 global regions
# to achieve medium-term (12-month) scalability targets:
# - 1,000 sustained RPS, 5,000 peak RPS
# - 10,000 concurrent agents
# - 100M actions storage
# - 10+ regions

set -e

# Configuration
IMAGE="nethical:latest"
DATA_BASE="/data/nethical"
CONFIG_DIR="./config"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Nethical Medium-Term Deployment${NC}"
echo -e "${BLUE}10 Regions - 1,000 RPS - 10,000 Agents${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed"
    exit 1
fi

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory not found: $CONFIG_DIR"
    exit 1
fi

# Build Docker image if needed
if [[ "$(docker images -q $IMAGE 2> /dev/null)" == "" ]]; then
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker build -t $IMAGE .
fi

# Define regions to deploy
declare -a REGIONS=(
    "us-east-1:8001"
    "us-west-2:8004"
    "eu-west-1:8002"
    "eu-central-1:8007"
    "ap-south-1:8003"
    "ap-northeast-1:8009"
    "ap-southeast-1:8010"
    "sa-east-1:8006"
    "ca-central-1:8005"
    "me-south-1:8008"
)

# Function to deploy a single region
deploy_region() {
    local region=$1
    local port=$2
    local container_name="nethical-$region"
    local config_file="$CONFIG_DIR/$region.env"
    local data_dir="$DATA_BASE/$region"
    
    echo -e "${YELLOW}Deploying $region...${NC}"
    
    # Check if config exists
    if [ ! -f "$config_file" ]; then
        echo -e "  ❌ Config not found: $config_file"
        return 1
    fi
    
    # Create data directory
    mkdir -p "$data_dir"
    mkdir -p "$data_dir/hot"
    mkdir -p "$data_dir/warm"
    mkdir -p "$data_dir/cold"
    
    # Stop existing container if running
    if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        echo -e "  Stopping existing container..."
        docker stop $container_name >/dev/null 2>&1 || true
        docker rm $container_name >/dev/null 2>&1 || true
    fi
    
    # Deploy container
    docker run -d \
        --name $container_name \
        --env-file $config_file \
        -p $port:8000 \
        -v $data_dir:/data/nethical \
        --restart unless-stopped \
        --health-cmd="curl -f http://localhost:8000/health || exit 1" \
        --health-interval=30s \
        --health-timeout=10s \
        --health-retries=3 \
        $IMAGE >/dev/null
    
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} $region deployed on port $port"
    else
        echo -e "  ❌ Failed to deploy $region"
        return 1
    fi
}

# Deploy Phase 1: Core Regions (Original 3)
echo -e "\n${BLUE}Phase 1: Deploying Core Regions (3)${NC}"
deploy_region "us-east-1" "8001"
deploy_region "eu-west-1" "8002"
deploy_region "ap-south-1" "8003"

# Deploy Phase 2: Americas Expansion
echo -e "\n${BLUE}Phase 2: Deploying Americas Expansion (3)${NC}"
deploy_region "us-west-2" "8004"
deploy_region "ca-central-1" "8005"
deploy_region "sa-east-1" "8006"

# Deploy Phase 3: Europe & Middle East
echo -e "\n${BLUE}Phase 3: Deploying Europe & Middle East (2)${NC}"
deploy_region "eu-central-1" "8007"
deploy_region "me-south-1" "8008"

# Deploy Phase 4: Asia-Pacific Expansion
echo -e "\n${BLUE}Phase 4: Deploying Asia-Pacific Expansion (2)${NC}"
deploy_region "ap-northeast-1" "8009"
deploy_region "ap-southeast-1" "8010"

# Wait for health checks
echo -e "\n${YELLOW}Waiting for health checks...${NC}"
sleep 10

# Verify deployments
echo -e "\n${BLUE}Verifying Deployments${NC}"
echo

deployed=0
failed=0

for region_port in "${REGIONS[@]}"; do
    IFS=':' read -r region port <<< "$region_port"
    container_name="nethical-$region"
    
    # Check container status
    status=$(docker inspect -f '{{.State.Status}}' $container_name 2>/dev/null || echo "not-found")
    health=$(docker inspect -f '{{.State.Health.Status}}' $container_name 2>/dev/null || echo "none")
    
    if [ "$status" == "running" ]; then
        echo -e "  ${GREEN}✓${NC} $region: running (health: $health) - http://localhost:$port"
        ((deployed++))
    else
        echo -e "  ❌ $region: $status"
        ((failed++))
    fi
done

# Summary
echo
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Deployment Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total regions: ${#REGIONS[@]}"
echo -e "${GREEN}Deployed: $deployed${NC}"
if [ $failed -gt 0 ]; then
    echo -e "❌ Failed: $failed"
fi
echo

# Display capacity
echo -e "${BLUE}System Capacity (Medium-Term Targets)${NC}"
echo -e "  Sustained RPS: ${deployed} × 100 = ${GREEN}$(($deployed * 100)) RPS${NC} (target: 1,000)"
echo -e "  Peak RPS: ${deployed} × 500 = ${GREEN}$(($deployed * 500)) RPS${NC} (target: 5,000)"
echo -e "  Concurrent Agents: ${deployed} × 1,000 = ${GREEN}$(($deployed * 1000))${NC} (target: 10,000)"
echo -e "  Storage: ${deployed} × 10M = ${GREEN}$(($deployed * 10))M actions${NC} (target: 100M)"
echo

if [ $deployed -eq ${#REGIONS[@]} ]; then
    echo -e "${GREEN}✓ All regions deployed successfully!${NC}"
    echo
    echo "Next steps:"
    echo "  1. Configure global load balancer"
    echo "  2. Set up monitoring dashboards"
    echo "  3. Run validation tests:"
    echo "     pytest tests/test_medium_term_scalability.py -v"
    echo "  4. Configure Redis cluster for distributed caching"
    echo "  5. Set up TimescaleDB for time-series storage"
    exit 0
else
    echo -e "${YELLOW}⚠ Some regions failed to deploy${NC}"
    echo "Check logs with: docker logs nethical-<region>"
    exit 1
fi
