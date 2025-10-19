#!/bin/bash
# Complete Nginx Fix Script

echo "🔧 Complete Nginx Fix..."

# Stop Nginx
sudo systemctl stop nginx

# Remove broken symlinks
sudo rm -f /etc/nginx/sites-enabled/default

# Backup broken config
sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.broken

# Create simple nginx.conf
sudo tee /etc/nginx/nginx.conf << 'EOF'
user www-data;
worker_processes auto;
pid /run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;
}
EOF

# Create simple default site
sudo tee /etc/nginx/sites-available/default << 'EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    
    root /var/www/html;
    index index.html index.htm index.nginx-debian.html;
    
    server_name _;
    
    location / {
        try_files $uri $uri/ =404;
    }
}
EOF

# Create correct symlink
sudo ln -s /etc/nginx/sites-available/default /etc/nginx/sites-enabled/

# Test configuration
echo "🧪 Testing Nginx configuration..."
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "✅ Configuration test passed!"
    
    # Start Nginx
    echo "🚀 Starting Nginx..."
    sudo systemctl start nginx
    
    # Check status
    echo "📊 Nginx Status:"
    sudo systemctl status nginx --no-pager -l
    
    echo ""
    echo "✅ Nginx should now be running!"
    echo "Test: curl http://localhost"
else
    echo "❌ Configuration test failed!"
    echo "Check: sudo nginx -t"
fi
