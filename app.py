"""
Versión optimizada para Render.com
NO requiere input() interactivo
Servidor HTTP siempre activo
"""
import os
os.environ['MPLBACKEND'] = 'Agg'

import sys
import time
import threading
import http.server
import socketserver

# Import todo del bot principal
sys.path.insert(0, os.path.dirname(__file__))
from crypto_pivot_v4_5_hybrid import (
    analyze_asset_full,
    educational_content,
    build_dashboard_hybrid,
    CONFIG,
    success,
    info,
    warn,
    err
)

# Puerto para Render
PORT = int(os.environ.get("PORT", 10000))

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Handler HTTP con logging mejorado"""
    
    def log_message(self, format, *args):
        info(f"{self.address_string()} - {format % args}")
    
    def do_GET(self):
        # Redirect root to dashboard
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/dashboard.html')
            self.end_headers()
            return
        
        return http.server.SimpleHTTPRequestHandler.do_GET(self)


def generate_dashboard():
    """Genera el dashboard una vez"""
    try:
        info("🔄 Generando dashboard...")
        
        # Analizar BTC
        info("📊 Analizando BTC...")
        btc_data = analyze_asset_full("BTC")
        
        # Analizar ETH
        info("📊 Analizando ETH...")
        eth_data = analyze_asset_full("ETH")
        
        # Contenido educativo
        edu = educational_content()
        
        # Construir dashboard
        info("🎨 Construyendo dashboard HTML...")
        build_dashboard_hybrid(btc_data, eth_data, edu, CONFIG["OUTPUT_DIR"])
        
        success(f"✅ Dashboard actualizado - BTC: {btc_data['geometric_bias']} ({btc_data['synthesis']['score']}/4⭐)")
        
        if btc_data.get('context_alert', {}).get('has_alert'):
            warn(f"⚠ Alerta activa: {btc_data['context_alert']['alert_type']}")
        
    except Exception as e:
        err(f"❌ Error generando dashboard: {str(e)}")
        import traceback
        traceback.print_exc()


def regenerate_loop():
    """Loop infinito que regenera el dashboard cada hora"""
    # Primera generación
    generate_dashboard()
    
    # Loop cada hora
    while True:
        try:
            info(f"⏰ Próxima actualización en 60 minutos...")
            time.sleep(3600)  # 60 minutos
            generate_dashboard()
        except Exception as e:
            err(f"Error en loop de regeneración: {str(e)}")
            time.sleep(300)  # 5 minutos antes de reintentar


def main():
    """Punto de entrada principal para Render"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  🔷 CRYPTO PIVOT ANALYZER v4.5 — RENDER DEPLOY                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
    
    # Crear directorios
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    os.makedirs(CONFIG["CACHE_DIR"], exist_ok=True)
    
    # Cambiar a directorio de salida
    os.chdir(CONFIG["OUTPUT_DIR"])
    
    # Thread para regeneración automática
    regen_thread = threading.Thread(target=regenerate_loop, daemon=True)
    regen_thread.start()
    
    # Esperar a que se genere el primer dashboard
    info("⏳ Esperando primera generación del dashboard...")
    time.sleep(5)  # Dar tiempo al thread para iniciar
    
    # Iniciar servidor HTTP
    Handler = DashboardHandler
    
    try:
        with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
            success(f"✓ Servidor HTTP iniciado en puerto {PORT}")
            success(f"✓ Dashboard disponible en http://0.0.0.0:{PORT}/dashboard.html")
            info("🔄 Auto-actualización cada 60 minutos")
            warn("⚠ Presiona Ctrl+C para detener")
            
            # Servir indefinidamente
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        warn("\n👋 Servidor detenido por usuario")
    except Exception as e:
        err(f"💥 Error fatal: {str(e)}")
        raise


if __name__ == "__main__":
    main()
