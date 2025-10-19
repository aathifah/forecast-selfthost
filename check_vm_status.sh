#!/bin/bash
# VM Status Check Script

echo "🔍 Checking VM Status..."
echo "========================"

# Check SSH service
echo "📡 SSH Service Status:"
sudo systemctl status ssh --no-pager -l

echo ""
echo "🌐 Network Interfaces:"
ip addr show

echo ""
echo "🔌 Listening Ports:"
sudo netstat -tlnp | grep -E ":(22|8080|80|443)"

echo ""
echo "📊 VM Resources:"
echo "CPU: $(nproc) cores"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Disk: $(df -h / | awk 'NR==2{print $2}')"

echo ""
echo "🐳 Docker Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "Docker not running"

echo ""
echo "✅ VM is ready for SSH connection!"
echo "Try: ssh ubuntu@127.0.0.1 -p 2222"

