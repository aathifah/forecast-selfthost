#!/bin/bash

# 🚀 Script Deployment Otomatis untuk Self-Hosting ML Forecast App
# Author: AI Assistant
# Description: Automated deployment script dengan backup dan rollback

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
BACKUP_DIR="/home/$USER/backups"
LOG_DIR="/home/$USER/logs"
DEPLOYMENT_LOG="$LOG_DIR/deployment.log"

# Create directories
mkdir -p $BACKUP_DIR $LOG_DIR

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $DEPLOYMENT_LOG
}

# Backup function
backup_app() {
    print_header "Creating Backup"
    local backup_name="${APP_NAME}_backup_$(date +%Y%m%d_%H%M%S)"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    log "Creating backup: $backup_name"
    
    # Stop current app
    docker-compose -f docker-compose.selfhost.yml down 2>/dev/null || true
    
    # Create backup
    mkdir -p $backup_path
    cp -r . $backup_path/
    
    # Remove unnecessary files from backup
    rm -rf $backup_path/node_modules $backup_path/.git $backup_path/__pycache__
    
    # Compress backup
    tar -czf "$backup_path.tar.gz" -C $BACKUP_DIR $backup_name
    rm -rf $backup_path
    
    log "Backup created: $backup_path.tar.gz"
    echo $backup_path.tar.gz
}

# Rollback function
rollback_app() {
    print_header "Rolling Back"
    
    # Get latest backup
    local latest_backup=$(ls -t $BACKUP_DIR/*.tar.gz | head -n1)
    
    if [ -z "$latest_backup" ]; then
        print_error "No backup found for rollback!"
        exit 1
    fi
    
    log "Rolling back to: $latest_backup"
    
    # Stop current app
    docker-compose -f docker-compose.selfhost.yml down 2>/dev/null || true
    
    # Extract backup
    local temp_dir="/tmp/rollback_$(date +%s)"
    mkdir -p $temp_dir
    tar -xzf $latest_backup -C $temp_dir
    
    # Replace current files
    cp -r $temp_dir/* .
    rm -rf $temp_dir
    
    # Start app
    docker-compose -f docker-compose.selfhost.yml up -d
    
    log "Rollback completed"
    print_status "Rollback completed successfully!"
}

# Health check function
health_check() {
    print_header "Health Check"
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8080/api/health >/dev/null 2>&1; then
            print_status "Health check passed!"
            return 0
        fi
        
        print_warning "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 10
        ((attempt++))
    done
    
    print_error "Health check failed after $max_attempts attempts!"
    return 1
}

# Deploy function
deploy_app() {
    print_header "Deploying ML Forecast App"
    
    # Create backup
    local backup_file=$(backup_app)
    
    # Pull latest changes (if using git)
    if [ -d ".git" ]; then
        print_status "Pulling latest changes..."
        git pull origin main || print_warning "Git pull failed, continuing with local changes..."
    fi
    
    # Build new image
    print_status "Building Docker image..."
    docker build -f Dockerfile.selfhost -t $APP_NAME:latest .
    
    # Stop old containers
    print_status "Stopping old containers..."
    docker-compose -f docker-compose.selfhost.yml down
    
    # Start new containers
    print_status "Starting new containers..."
    docker-compose -f docker-compose.selfhost.yml up -d
    
    # Wait for containers to start
    sleep 30
    
    # Health check
    if health_check; then
        print_status "Deployment successful!"
        
        # Cleanup old images
        docker image prune -f
        
        # Keep only last 5 backups
        ls -t $BACKUP_DIR/*.tar.gz | tail -n +6 | xargs rm -f 2>/dev/null || true
        
        log "Deployment completed successfully"
    else
        print_error "Deployment failed! Rolling back..."
        rollback_app
        exit 1
    fi
}

# Update dependencies
update_dependencies() {
    print_header "Updating Dependencies"
    
    # Update system packages
    print_status "Updating system packages..."
    sudo apt update && sudo apt upgrade -y
    
    # Update Docker
    print_status "Updating Docker..."
    sudo apt install -y docker.io docker-compose
    
    # Update Python packages
    print_status "Updating Python packages..."
    pip install --upgrade pip
    pip install -r requirements.txt --upgrade
    
    log "Dependencies updated"
}

# Monitor function
monitor_app() {
    print_header "Monitoring Application"
    
    while true; do
        clear
        echo "📊 ML Forecast App Monitor - $(date)"
        echo "=================================="
        
        # Docker status
        echo "🐳 Docker Containers:"
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        
        echo ""
        echo "🌐 Application Health:"
        if curl -s http://localhost:8080/api/health | jq . 2>/dev/null; then
            echo "✅ Application is healthy"
        else
            echo "❌ Application health check failed"
        fi
        
        echo ""
        echo "💾 System Resources:"
        echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')"
        echo "Memory: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')"
        echo "Disk: $(df -h / | awk 'NR==2{printf "%s", $5}')"
        
        echo ""
        echo "📈 Application Logs (last 10 lines):"
        docker logs --tail 10 $APP_NAME 2>/dev/null || echo "No logs available"
        
        echo ""
        echo "Press Ctrl+C to exit monitoring"
        sleep 10
    done
}

# Maintenance function
maintenance_mode() {
    print_header "Maintenance Mode"
    
    local action=$1
    
    case $action in
        "enable")
            print_status "Enabling maintenance mode..."
            
            # Create maintenance page
            cat > static/maintenance.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Maintenance Mode</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        .container { max-width: 600px; margin: 0 auto; }
        h1 { color: #e74c3c; }
        p { color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 Maintenance Mode</h1>
        <p>Website sedang dalam mode maintenance.</p>
        <p>Kami akan kembali online dalam beberapa saat.</p>
        <p>Terima kasih atas kesabaran Anda.</p>
    </div>
</body>
</html>
EOF
            
            # Update nginx to show maintenance page
            sudo cp nginx.conf /etc/nginx/nginx.conf
            sudo sed -i 's/proxy_pass http:\/\/ml_forecast;/return 503;/' /etc/nginx/nginx.conf
            sudo sed -i '/return 503;/a\        error_page 503 @maintenance;\n        location @maintenance {\n            root /home/'$USER'/railways-web-forecast/static;\n            rewrite ^(.*)$ /maintenance.html break;\n        }' /etc/nginx/nginx.conf
            
            sudo nginx -s reload
            log "Maintenance mode enabled"
            ;;
        "disable")
            print_status "Disabling maintenance mode..."
            
            # Restore normal nginx config
            sudo cp nginx.conf /etc/nginx/nginx.conf
            sudo nginx -s reload
            
            # Remove maintenance page
            rm -f static/maintenance.html
            log "Maintenance mode disabled"
            ;;
        *)
            print_error "Usage: $0 maintenance [enable|disable]"
            exit 1
            ;;
    esac
}

# Main menu
show_menu() {
    echo ""
    print_header "ML Forecast App Deployment Manager"
    echo "1. Deploy Application"
    echo "2. Rollback Application"
    echo "3. Update Dependencies"
    echo "4. Monitor Application"
    echo "5. Enable Maintenance Mode"
    echo "6. Disable Maintenance Mode"
    echo "7. Health Check"
    echo "8. View Logs"
    echo "9. Exit"
    echo ""
}

# View logs function
view_logs() {
    print_header "Application Logs"
    
    echo "📋 Recent deployment logs:"
    tail -20 $DEPLOYMENT_LOG
    
    echo ""
    echo "🐳 Docker container logs:"
    docker logs --tail 50 $APP_NAME 2>/dev/null || echo "No container logs available"
    
    echo ""
    echo "🌐 Nginx logs:"
    sudo tail -20 /var/log/nginx/access.log
    sudo tail -20 /var/log/nginx/error.log
}

# Main script logic
main() {
    case $1 in
        "deploy")
            deploy_app
            ;;
        "rollback")
            rollback_app
            ;;
        "update")
            update_dependencies
            ;;
        "monitor")
            monitor_app
            ;;
        "maintenance")
            maintenance_mode $2
            ;;
        "health")
            health_check
            ;;
        "logs")
            view_logs
            ;;
        *)
            show_menu
            read -p "Pilih opsi (1-9): " choice
            
            case $choice in
                1) deploy_app ;;
                2) rollback_app ;;
                3) update_dependencies ;;
                4) monitor_app ;;
                5) maintenance_mode "enable" ;;
                6) maintenance_mode "disable" ;;
                7) health_check ;;
                8) view_logs ;;
                9) exit 0 ;;
                *) print_error "Opsi tidak valid!" ;;
            esac
            ;;
    esac
}

# Run main function
main "$@"
