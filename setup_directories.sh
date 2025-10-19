#!/bin/bash
# Setup script untuk membuat directory dan file yang diperlukan

echo "🚀 Setting up ML Forecast App directories..."

# Buat directory yang diperlukan
mkdir -p data logs ssl static

# Set permissions
chmod 755 data logs ssl static

# Buat file .env jika belum ada
if [ ! -f .env ]; then
    cat > .env << EOF
PORT=8080
PYTHONUNBUFFERED=1
FORECAST_N_JOBS=2
FORECAST_PROGRESS=1
FORECAST_DEBUG=0
MAX_DATA_ROWS=1000000
MAX_DATA_MONTHS=12
EOF
    echo "✅ Created .env file"
fi

# Buat file placeholder untuk SSL (akan diisi nanti)
if [ ! -f ssl/placeholder ]; then
    echo "# SSL certificates will be placed here" > ssl/placeholder
    echo "✅ Created SSL directory"
fi

echo "✅ Setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Run: docker-compose -f docker-compose.simple.yml up -d"
echo "2. Check logs: docker logs ml-forecast-app"
echo "3. Test app: curl http://localhost:8080/api/health"
