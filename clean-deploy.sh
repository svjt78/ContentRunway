#!/bin/bash

# ContentRunway Clean Deployment Script
# Complete deployment with clean environment to avoid API key caching issues

echo "🧹 Clean deployment of ContentRunway..."

# Stop and remove everything
echo "📦 Stopping containers and removing volumes..."
docker-compose down -v

# Rebuild images
echo "🔨 Rebuilding Docker images..."
docker-compose build

# Start with clean environment
echo "🚀 Starting with fresh environment..."
env -i bash -c 'source .env && docker-compose up -d'

echo "✅ Clean deployment completed!"
echo "💡 Verify API key: docker exec contentrunway-langgraph-worker-1 env | grep OPENAI"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"