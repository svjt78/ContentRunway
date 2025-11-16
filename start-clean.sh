#!/bin/bash

# ContentRunway Clean Start Script
# Ensures Docker Compose uses .env file values instead of system environment

echo "🧹 Starting ContentRunway with clean environment..."

# Stop any running containers
echo "📦 Stopping existing containers..."
docker-compose down

# Start with clean environment that only uses .env file
echo "🚀 Starting with fresh environment from .env file..."
env -i bash -c 'source .env && docker-compose up -d'

echo "✅ ContentRunway started successfully!"
echo "💡 To verify API key: docker exec contentrunway-langgraph-worker-1 env | grep OPENAI"