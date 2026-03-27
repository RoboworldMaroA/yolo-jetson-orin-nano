#!/bin/bash

# --- KONFIGURACJA ---
USB_MOUNT="/media/marek/dysk_usb"
USB_GIT="$USB_MOUNT/git_backup_projects/yolo_app.git"
USB_DATA="$USB_MOUNT/full_project_copy/"
PROJECT_DIR="/home/maro/yolo_app/"

echo "=== ROZPOCZYNAM BACKUP HYBRYDOWY ==="

# 1. POBIERANIE OPISU ZMIAN
echo "Wpisz opis zmian (commit message):"
read message
if [ -z "$message" ]; then
  message="Backup: $(date +'%Y-%m-%d %H:%M')"
fi

# 2. LOKALNY COMMIT (Zawsze robimy commit przed wysyłką)
git add .
git commit -m "$message"

echo "-----------------------------------"
echo "🚀 [1/3] Wysyłam KOD na GitHub..."
git push origin main

# 3. SPRAWDZENIE CZY DYSK USB JEST PODPIĘTY
if mountpoint -q "$USB_MOUNT"; then
    echo "-----------------------------------"
    echo "✅ Dysk USB wykryty. Kontynuuję backup na nośnik zewnętrzny."
    
    echo "🚀 [2/3] Wysyłam KOD na USB (Git Repo)..."
    git push local-usb main

    echo "📂 [3/3] Synchronizuję DANE i MODELE na USB (rsync)..."
    mkdir -p "$USB_DATA"
rsync -rtvz --progress --exclude='__pycache__/' --exclude='yolo-app/' --exclude='parkingVenv/' --exclude='*.pyc' "$PROJECT_DIR" "$USB_DATA"
    
    echo "-----------------------------------"
    echo "✅ PEŁNY BACKUP ZAKOŃCZONY (GitHub + USB)."
else
    echo "-----------------------------------"
    echo "⚠️ OSTRZEŻENIE: Dysk USB NIE jest podpięty (brak mountpoint)."
    echo "Kod został wysłany TYLKO na GitHub."
    echo "Podepnij dysk i uruchom skrypt ponownie, aby zabezpieczyć dane i modele."
    echo "-----------------------------------"
fi

echo "Gotowe."
