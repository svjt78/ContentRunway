#!/bin/bash

# ContentRunway Full Reset Script
# Nuclear option: complete system reset when everything goes wrong

echo "💥 Full reset of ContentRunway..."

# Stop everything
echo "📦 Stopping all containers and removing volumes..."
docker-compose down -v

# Clear Docker cache
echo "🧹 Clearing Docker system cache..."
docker system prune -f

# Remove Python cache from mounted volumes
echo "🐍 Clearing Python cache from volumes..."
find ./langgraph -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find ./backend -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Rebuild and start clean
echo "🔨 Rebuilding images without cache..."
docker-compose build --no-cache

echo "🚀 Starting with fresh environment..."
env -i bash -c 'source .env && docker-compose up -d'

echo "✅ Full reset completed!"
echo "💡 Verify API key: docker exec contentrunway-langgraph-worker-1 env | grep OPENAI"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"