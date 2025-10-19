#!/bin/bash
# Fix permissions script untuk VM Ubuntu

echo "🔧 Fixing permissions for ML Forecast App directories..."

# Fix ownership dan permissions
sudo chown -R ubuntu:ubuntu data logs ssl
sudo chmod -R 755 data logs ssl

# Buat file placeholder untuk SSL
echo "# SSL certificates will be placed here" | sudo tee ssl/placeholder
sudo chown ubuntu:ubuntu ssl/placeholder

echo "✅ Permissions fixed!"
echo ""
echo "📋 Next steps:"
echo "1. Clean orphan containers: docker-compose -f docker-compose.selfhost.yml down --remove-orphans"
echo "2. Start fresh: docker-compose -f docker-compose.selfhost.yml up -d"
echo "3. Check status: docker ps"
echo "4. Test app: curl http://localhost:8080/api/health"
