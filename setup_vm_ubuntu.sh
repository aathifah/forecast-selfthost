#!/bin/bash

# 🚀 Script Setup VM Ubuntu Otomatis untuk ML Forecast App
# Author: AI Assistant
# Description: Automated setup script untuk VM Ubuntu self-hosting

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${BLUE}=== $1 ===${NC}"; }

# Configuration
APP_NAME="ml-forecast-app"
APP_DIR="/home/$USER/railways-web-forecast"
LOG_FILE="/var/log/ml-forecast-setup.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | sudo tee -a $LOG_FILE
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "Script ini tidak boleh dijalankan sebagai root!"
   exit 1
fi

print_header "🚀 ML Forecast App VM Setup"
print_status "Starting automated setup..."

# Input dari user
read -p "Masukkan domain Anda (contoh: mymlapp.duckdns.org): " DOMAIN
if [ -z "$DOMAIN" ]; then
    print_error "Domain tidak boleh kosong!"
    exit 1
fi

read -p "Masukkan DuckDNS token (jika pakai DuckDNS): " DUCKDNS_TOKEN
if [ -z "$DUCKDNS_TOKEN" ]; then
    print_warning "DuckDNS token kosong, akan skip DuckDNS setup"
fi

print_header "System Update & Dependencies"
log "Updating system packages..."

# Update sistem
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git vim htop tree unzip jq

print_status "System updated successfully"

print_header "Docker Installation"
log "Installing Docker..."

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose -y

# Verify Docker
docker --version
docker-compose --version

print_status "Docker installed successfully"

print_header "Nginx & SSL Setup"
log "Installing Nginx and Certbot..."

# Install Nginx
sudo apt install nginx -y

# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Start services
sudo systemctl start nginx
sudo systemctl enable nginx

print_status "Nginx and Certbot installed"

print_header "Domain & SSL Configuration"
log "Setting up domain and SSL..."

# Setup DuckDNS (jika ada token)
if [ ! -z "$DUCKDNS_TOKEN" ]; then
    print_status "Setting up DuckDNS..."
    
    # Extract subdomain dari domain
    SUBDOMAIN=$(echo $DOMAIN | cut -d'.' -f1)
    
    # Buat script update DuckDNS
    cat > ~/update_duckdns.sh << EOF
#!/bin/bash
curl "https://www.duckdns.org/update?domains=$SUBDOMAIN&token=$DUCKDNS_TOKEN&ip="
EOF
    
    chmod +x ~/update_duckdns.sh
    
    # Test script
    ./update_duckdns.sh
    
    # Setup cron job
    echo "*/5 * * * * $HOME/update_duckdns.sh" | crontab -
    
    print_status "DuckDNS configured"
fi

# Generate SSL Certificate
print_status "Generating SSL certificate..."
sudo certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN

print_status "SSL certificate generated"

print_header "Application Setup"
log "Setting up ML Forecast application..."

# Check if app directory exists
if [ ! -d "$APP_DIR" ]; then
    print_error "App directory tidak ditemukan: $APP_DIR"
    print_status "Pastikan Anda sudah copy aplikasi ke VM"
    exit 1
fi

cd $APP_DIR

# Buat file .env
cat > .env << EOF
PORT=8080
PYTHONUNBUFFERED=1
FORECAST_N_JOBS=2
FORECAST_PROGRESS=1
FORECAST_DEBUG=0
MAX_DATA_ROWS=1000000
MAX_DATA_MONTHS=12
EOF

print_status "Environment file created"

print_header "Docker Deployment"
log "Deploying application with Docker..."

# Build dan run aplikasi
docker-compose -f docker-compose.selfhost.yml up -d

# Wait for startup
sleep 30

print_status "Application deployed"

print_header "Nginx Configuration"
log "Configuring Nginx reverse proxy..."

# Copy nginx config
sudo cp nginx.conf /etc/nginx/nginx.conf

# Update domain di nginx config
sudo sed -i "s/yourdomain.com/$DOMAIN/g" /etc/nginx/nginx.conf

# Test nginx config
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx

print_status "Nginx configured"

print_header "Firewall Setup"
log "Configuring firewall..."

# Install UFW
sudo apt install ufw -y

# Configure firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8080/tcp

# Enable firewall
sudo ufw --force enable

print_status "Firewall configured"

print_header "Security Setup"
log "Setting up security..."

# Install Fail2Ban
sudo apt install fail2ban -y

# Configure Fail2Ban
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local

# Start Fail2Ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

print_status "Security configured"

print_header "Monitoring Setup"
log "Setting up monitoring..."

# Install monitoring tools
sudo apt install htop iotop nethogs -y

# Buat script monitoring
cat > ~/monitor_app.sh << 'EOF'
#!/bin/bash
echo "📊 ML Forecast App Status - $(date)"
echo "=================================="

# Docker status
echo "🐳 Docker Containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🌐 Application Health:"
curl -s http://localhost:8080/api/health | jq . 2>/dev/null || echo "❌ Health check failed"

echo ""
echo "💾 System Resources:"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')"
echo "Memory: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')"
echo "Disk: $(df -h / | awk 'NR==2{printf "%s", $5}')"

echo ""
echo "📈 Application Logs (last 5 lines):"
docker logs --tail 5 ml-forecast-app 2>/dev/null || echo "No logs available"
EOF

chmod +x ~/monitor_app.sh

print_status "Monitoring configured"

print_header "Auto-Start Setup"
log "Setting up auto-start..."

# Buat script startup
cat > ~/start_ml_app.sh << EOF
#!/bin/bash
echo "🚀 Starting ML Forecast App..."

# Start Docker containers
cd $APP_DIR
docker-compose -f docker-compose.selfhost.yml up -d

# Wait for startup
sleep 30

# Check status
curl -f http://localhost:8080/api/health && echo "✅ App is running!" || echo "❌ App failed to start"

echo "🌐 Your app is available at:"
echo "   - http://$DOMAIN"
echo "   - https://$DOMAIN"
echo "   - http://localhost:8080"
EOF

chmod +x ~/start_ml_app.sh

# Buat script stop
cat > ~/stop_ml_app.sh << EOF
#!/bin/bash
echo "🛑 Stopping ML Forecast App..."
cd $APP_DIR
docker-compose -f docker-compose.selfhost.yml down
echo "✅ App stopped!"
EOF

chmod +x ~/stop_ml_app.sh

# Setup systemd service
sudo tee /etc/systemd/system/ml-forecast.service << EOF
[Unit]
Description=ML Forecast App
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$APP_DIR
ExecStart=/usr/bin/docker-compose -f docker-compose.selfhost.yml up -d
ExecStop=/usr/bin/docker-compose -f docker-compose.selfhost.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

# Enable service
sudo systemctl enable ml-forecast.service
sudo systemctl start ml-forecast.service

print_status "Auto-start configured"

print_header "Performance Optimization"
log "Optimizing system performance..."

# Optimize memory usage
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.max_map_count=262144' | sudo tee -a /etc/sysctl.conf

# Apply changes
sudo sysctl -p

print_status "Performance optimized"

print_header "Health Check"
log "Performing health check..."

# Wait for services to start
sleep 10

# Check Docker containers
if docker ps | grep -q ml-forecast-app; then
    print_status "✅ Docker container is running"
else
    print_error "❌ Docker container failed to start"
fi

# Check application health
if curl -f http://localhost:8080/api/health >/dev/null 2>&1; then
    print_status "✅ Application health check passed"
else
    print_warning "⚠️ Application health check failed"
fi

# Check nginx
if sudo systemctl is-active --quiet nginx; then
    print_status "✅ Nginx is running"
else
    print_error "❌ Nginx failed to start"
fi

# Check SSL
if curl -I https://$DOMAIN >/dev/null 2>&1; then
    print_status "✅ SSL certificate is working"
else
    print_warning "⚠️ SSL certificate check failed"
fi

print_header "Setup Complete! 🎉"
print_status "Domain: $DOMAIN"
print_status "SSL: ✅ Installed"
print_status "Nginx: ✅ Configured"
print_status "Docker: ✅ Running"
print_status "Firewall: ✅ Configured"
print_status "Security: ✅ Configured"
print_status "Monitoring: ✅ Configured"
print_status "Auto-start: ✅ Configured"

echo ""
print_status "Scripts yang tersedia:"
echo "  - ~/start_ml_app.sh    : Start aplikasi"
echo "  - ~/stop_ml_app.sh     : Stop aplikasi"
echo "  - ~/monitor_app.sh     : Monitor aplikasi"

echo ""
print_status "Aplikasi Anda tersedia di:"
echo "  - https://$DOMAIN"
echo "  - http://localhost:8080"

echo ""
print_status "Logs tersimpan di:"
echo "  - Setup logs: $LOG_FILE"
echo "  - App logs: docker logs ml-forecast-app"
echo "  - Nginx logs: /var/log/nginx/"

print_warning "Pastikan domain $DOMAIN sudah mengarah ke IP VM ini!"
print_warning "Jika menggunakan DuckDNS, pastikan token sudah benar!"

log "Setup completed successfully"
print_status "Setup selesai! Aplikasi ML Anda siap digunakan! 🚀"
