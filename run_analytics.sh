#!/bin/bash
# Innov8 Ultra Analytics Dashboard Launcher
# Beltone Hackathon 2nd Edition

echo "🚀 Innov8 Ultra Analytics Dashboard"
echo "===================================="
echo ""
echo "Starting dashboard on port 8502..."
echo ""

cd "$(dirname "$0")"

# Kill any existing instances on this port
lsof -ti:8502 | xargs kill -9 2>/dev/null

# Launch the dashboard
streamlit run solver_analytics_dashboard.py --server.port 8502 --server.headless false

echo ""
echo "Dashboard is now running!"
echo "Open your browser to: http://localhost:8502"
echo ""
echo "Press Ctrl+C to stop the server"

