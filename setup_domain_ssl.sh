#!/bin/bash

# 🌐 Script Setup Domain dan SSL Gratis untuk Self-Hosting ML App
# Author: AI Assistant
# Description: Setup domain, SSL, dan reverse proxy untuk aplikasi ML

set -e

echo "🚀 Setting up domain dan SSL untuk ML Forecast App..."

# Colors untuk output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function untuk print dengan warna
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "Script ini tidak boleh dijalankan sebagai root!"
   exit 1
fi

# Input domain dari user
read -p "Masukkan domain Anda (contoh: mymlapp.com): " DOMAIN
if [ -z "$DOMAIN" ]; then
    print_error "Domain tidak boleh kosong!"
    exit 1
fi

print_header "Setup Domain: $DOMAIN"

# Install dependencies
print_status "Installing dependencies..."
sudo apt update
sudo apt install -y curl wget git nginx certbot python3-certbot-nginx

# Setup Nginx
print_status "Setting up Nginx..."
sudo systemctl enable nginx
sudo systemctl start nginx

# Create SSL directory
sudo mkdir -p /etc/nginx/ssl
sudo chmod 755 /etc/nginx/ssl

# Update nginx.conf dengan domain yang benar
print_status "Updating Nginx configuration..."
sudo cp nginx.conf /etc/nginx/nginx.conf
sudo sed -i "s/yourdomain.com/$DOMAIN/g" /etc/nginx/nginx.conf

# Test nginx configuration
sudo nginx -t

# Setup Let's Encrypt SSL
print_header "Setting up SSL Certificate"
print_status "Getting SSL certificate dari Let's Encrypt..."

# Stop nginx temporarily untuk certbot
sudo systemctl stop nginx

# Get certificate
sudo certbot certonly --standalone -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN

# Copy certificates ke nginx ssl directory
sudo cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem /etc/nginx/ssl/
sudo cp /etc/letsencrypt/live/$DOMAIN/privkey.pem /etc/nginx/ssl/
sudo chmod 644 /etc/nginx/ssl/fullchain.pem
sudo chmod 600 /etc/nginx/ssl/privkey.pem

# Start nginx
sudo systemctl start nginx

# Setup auto-renewal
print_status "Setting up SSL auto-renewal..."
echo "0 12 * * * /usr/bin/certbot renew --quiet --post-hook 'systemctl reload nginx'" | sudo crontab -

# Setup Cloudflare Tunnel (Optional)
print_header "Setup Cloudflare Tunnel (Optional)"
read -p "Apakah Anda ingin menggunakan Cloudflare Tunnel? (y/n): " USE_CLOUDFLARE

if [ "$USE_CLOUDFLARE" = "y" ] || [ "$USE_CLOUDFLARE" = "Y" ]; then
    print_status "Installing Cloudflare Tunnel..."
    
    # Download cloudflared
    wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i cloudflared-linux-amd64.deb
    rm cloudflared-linux-amd64.deb
    
    print_status "Login ke Cloudflare..."
    cloudflared tunnel login
    
    print_status "Creating tunnel..."
    cloudflared tunnel create ml-forecast-tunnel
    
    print_status "Setting up tunnel configuration..."
    cat > ~/.cloudflared/config.yml << EOF
tunnel: ml-forecast-tunnel
credentials-file: /home/$USER/.cloudflared/ml-forecast-tunnel.json

ingress:
  - hostname: $DOMAIN
    service: http://localhost:80
  - hostname: www.$DOMAIN
    service: http://localhost:80
  - service: http_status:404
EOF

    # Setup tunnel sebagai systemd service
    sudo cloudflared service install
    sudo systemctl enable cloudflared
    sudo systemctl start cloudflared
    
    print_status "Cloudflare Tunnel berhasil diinstall!"
fi

# Setup DuckDNS (Alternative)
print_header "Setup DuckDNS (Alternative)"
read -p "Apakah Anda ingin menggunakan DuckDNS sebagai backup? (y/n): " USE_DUCKDNS

if [ "$USE_DUCKDNS" = "y" ] || [ "$USE_DUCKDNS" = "Y" ]; then
    read -p "Masukkan DuckDNS subdomain (contoh: mymlapp): " DUCKDNS_SUBDOMAIN
    read -p "Masukkan DuckDNS token: " DUCKDNS_TOKEN
    
    # Setup DuckDNS update script
    cat > ~/update_duckdns.sh << EOF
#!/bin/bash
curl "https://www.duckdns.org/update?domains=$DUCKDNS_SUBDOMAIN&token=$DUCKDNS_TOKEN&ip="
EOF
    
    chmod +x ~/update_duckdns.sh
    
    # Add to crontab
    echo "*/5 * * * * $HOME/update_duckdns.sh" | crontab -
    
    print_status "DuckDNS berhasil diinstall!"
fi

# Setup firewall
print_header "Setting up Firewall"
print_status "Configuring UFW firewall..."
sudo ufw --force enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8080/tcp

# Create startup script
print_header "Creating startup script"
cat > ~/start_ml_app.sh << EOF
#!/bin/bash
echo "🚀 Starting ML Forecast App..."

# Start Docker containers
cd $(pwd)
docker-compose -f docker-compose.selfhost.yml up -d

# Check status
echo "📊 Checking application status..."
sleep 10
curl -f http://localhost:8080/api/health && echo "✅ App is running!" || echo "❌ App failed to start"

echo "🌐 Your app is available at:"
echo "   - http://$DOMAIN"
echo "   - https://$DOMAIN"
echo "   - http://localhost:8080"
EOF

chmod +x ~/start_ml_app.sh

# Create stop script
cat > ~/stop_ml_app.sh << EOF
#!/bin/bash
echo "🛑 Stopping ML Forecast App..."
cd $(pwd)
docker-compose -f docker-compose.selfhost.yml down
echo "✅ App stopped!"
EOF

chmod +x ~/stop_ml_app.sh

# Create monitoring script
cat > ~/monitor_ml_app.sh << EOF
#!/bin/bash
echo "📊 ML Forecast App Status:"
echo "=========================="

# Check Docker containers
echo "🐳 Docker Containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🌐 Application Health:"
curl -s http://localhost:8080/api/health | jq . || echo "❌ Health check failed"

echo ""
echo "💾 System Resources:"
echo "CPU Usage: \$(top -bn1 | grep "Cpu(s)" | awk '{print \$2}' | awk -F'%' '{print \$1}')"
echo "Memory Usage: \$(free | grep Mem | awk '{printf("%.1f%%", \$3/\$2 * 100.0)}')"
echo "Disk Usage: \$(df -h / | awk 'NR==2{printf "%s", \$5}')"
EOF

chmod +x ~/monitor_ml_app.sh

print_header "Setup Complete! 🎉"
print_status "Domain: $DOMAIN"
print_status "SSL: ✅ Installed"
print_status "Nginx: ✅ Configured"
print_status "Firewall: ✅ Configured"

echo ""
print_status "Scripts yang tersedia:"
echo "  - ~/start_ml_app.sh    : Start aplikasi"
echo "  - ~/stop_ml_app.sh     : Stop aplikasi"
echo "  - ~/monitor_ml_app.sh  : Monitor aplikasi"

echo ""
print_status "Next steps:"
echo "1. Jalankan: ~/start_ml_app.sh"
echo "2. Buka browser ke: https://$DOMAIN"
echo "3. Monitor dengan: ~/monitor_ml_app.sh"

print_warning "Pastikan domain $DOMAIN sudah mengarah ke IP server ini!"
print_warning "Jika menggunakan Cloudflare, pastikan proxy status adalah 'DNS only' (gray cloud)"
