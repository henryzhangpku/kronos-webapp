#!/bin/bash

# Kronos AI Web App - vast.ai Deployment Script
# This script automates the deployment of the Kronos web application on vast.ai

set -e  # Exit on any error

echo "🚀 Starting Kronos AI Web App Deployment on vast.ai"
echo "================================================="

# Configuration
IMAGE_NAME="kronos-webapp"
CONTAINER_NAME="kronos-app"
PORT=5000
REPO_URL="https://github.com/henryzhangpku/kronos-webapp.git"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_status "Checking prerequisites..."
if ! command_exists docker; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command_exists git; then
    print_error "Git is not installed. Please install Git first."
    exit 1
fi

print_success "Prerequisites check passed"

# Clone or update repository
if [ -d "kronos-webapp" ]; then
    print_status "Updating existing repository..."
    cd kronos-webapp
    git pull origin main
else
    print_status "Cloning repository..."
    git clone $REPO_URL
    cd kronos-webapp
fi

print_success "Repository ready"

# Stop and remove existing container if it exists
print_status "Cleaning up existing containers..."
if docker ps -a | grep -q $CONTAINER_NAME; then
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    print_success "Cleaned up existing container"
fi

# Remove existing image if it exists
if docker images | grep -q $IMAGE_NAME; then
    docker rmi $IMAGE_NAME 2>/dev/null || true
    print_success "Cleaned up existing image"
fi

# Build Docker image
print_status "Building Docker image..."
docker build -t $IMAGE_NAME .
if [ $? -eq 0 ]; then
    print_success "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Run container
print_status "Starting container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:5000 \
    --restart unless-stopped \
    $IMAGE_NAME

if [ $? -eq 0 ]; then
    print_success "Container started successfully"
else
    print_error "Failed to start container"
    exit 1
fi

# Wait a moment for the container to start
sleep 5

# Check if container is running
if docker ps | grep -q $CONTAINER_NAME; then
    print_success "Container is running"
else
    print_error "Container failed to start. Checking logs..."
    docker logs $CONTAINER_NAME
    exit 1
fi

# Get container logs to check for any startup issues
print_status "Checking application status..."
sleep 10

# Test if the application is responding
if command_exists curl; then
    if curl -f http://localhost:$PORT >/dev/null 2>&1; then
        print_success "Application is responding on port $PORT"
    else
        print_warning "Application might not be ready yet. Check logs with: docker logs $CONTAINER_NAME"
    fi
else
    print_warning "curl not available. Cannot test application response."
fi

echo ""
echo "🎉 Deployment completed!"
echo "================================================="
echo "Container Name: $CONTAINER_NAME"
echo "Image Name: $IMAGE_NAME"
echo "Port: $PORT"
echo ""
echo "📱 Application should be accessible at:"
echo "   http://localhost:$PORT"
echo "   http://$(hostname -I | awk '{print $1}'):$PORT"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker logs $CONTAINER_NAME"
echo "   Stop app: docker stop $CONTAINER_NAME"
echo "   Start app: docker start $CONTAINER_NAME"
echo "   Restart app: docker restart $CONTAINER_NAME"
echo "   Remove app: docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"
echo ""
echo "🔧 For vast.ai deployment:"
echo "   1. Upload this repository to your vast.ai instance"
echo "   2. Run this script: bash deploy.sh"
echo "   3. Access via the instance's public IP on port $PORT"
echo ""
print_success "Happy trading with Kronos AI! 📈"