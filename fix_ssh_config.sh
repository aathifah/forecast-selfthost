#!/bin/bash
# Fix SSH Configuration untuk VM Ubuntu

echo "🔧 Fixing SSH Configuration..."

# Backup original config
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

# Edit SSH config
sudo sed -i 's/#ListenAddress 0.0.0.0/ListenAddress 0.0.0.0/' /etc/ssh/sshd_config
sudo sed -i 's/#Port 22/Port 22/' /etc/ssh/sshd_config

# Add ListenAddress if not exists
if ! grep -q "ListenAddress 0.0.0.0" /etc/ssh/sshd_config; then
    echo "ListenAddress 0.0.0.0" | sudo tee -a /etc/ssh/sshd_config
fi

# Restart SSH
sudo systemctl restart ssh

# Check status
echo "📡 SSH Service Status:"
sudo systemctl status ssh --no-pager -l

echo ""
echo "🌐 SSH Listening Ports:"
sudo netstat -tlnp | grep :22

echo ""
echo "🔍 VM IP Addresses:"
ip addr show | grep "inet " | grep -v "127.0.0.1"

echo ""
echo "✅ SSH Configuration fixed!"
echo "Try SSH from Windows:"
echo "ssh ubuntu@127.0.0.1 -p 2222"
echo "or"
echo "ssh ubuntu@VM_IP_ADDRESS -p 22"

