# 🔷 Crypto Pivot Analyzer v4.5 Hybrid

Sistema avanzado de análisis técnico para Bitcoin y Ethereum que combina múltiples metodologías:
- **TIME Validation** (ICT): Análisis de formación de extremos por hora
- **DISTANCE Validation** (ICT): Evaluación de displacement vs histórico
- **P1/P2 Detection** (SMC): Detección de pivots semanales con Flip Risk
- **Pivots Tradicionales**: PP, R1, S1, R2, S2
- **Proyecciones de Precio**: Escenarios bull/bear con probabilidades
- **Alerta de Contexto**: Detección automática de conflictos entre timeframes
- **Sistema Educativo**: Explicaciones inline para cada métrica

## 🚀 Características

- ✅ Multi-asset (BTC + ETH)
- ✅ Multi-timeframe (Weekly + Daily + 4H)
- ✅ Auto-refresh cada 60 minutos
- ✅ Servidor web local integrado
- ✅ Dashboard interactivo con explicaciones educativas
- ✅ Score de decisión (0-4 estrellas)
- ✅ Detección de trampas alcistas/bajistas
- ✅ 12 meses de datos históricos 4H vía CCXT

## 📦 Instalación

```bash
# Clonar repositorio
git clone https://github.com/TU_USUARIO/crypto-pivot-analyzer.git
cd crypto-pivot-analyzer

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python crypto_pivot_v4_5_hybrid.py
```

## 🌐 Uso del Servidor Web

Cuando ejecutes el bot, te preguntará si quieres iniciar un servidor web:

```
¿Quieres iniciar un servidor web local para ver el dashboard? (s/n): s
```

Esto te permitirá:
- Ver el dashboard en `http://localhost:8080/dashboard.html`
- Acceder desde tu celular/tablet en la misma WiFi
- Mantener el dashboard actualizado automáticamente cada hora

## 📊 Estructura del Dashboard

### 1. Precio en Vivo & Niveles
- OHLC actual
- Pivots tradicionales (PP, R1, S1, R2, S2)
- Weekly/Monthly Open
- Geometric Bias

### 2. Proyecciones de Precio
- **Movimiento Actual**: Detecta dirección del precio
- Escenarios alcista/bajista con confidence %
- Probabilidad de REVERSIÓN vs CONTINUACIÓN

### 3. ⚠ Alerta de Contexto (NUEVO)
Detecta automáticamente:
- Rebote dentro de estructura bajista
- Pullback dentro de estructura alcista
- Trampas alcistas/bajistas sin confirmar
- Alineación completa de señales

### 4. TIME Validation
- ¿El high/low formado a hora H típicamente HOLDS o TAKEN?
- Warning de formación temprana
- % histórico de sostenimiento

### 5. DISTANCE Validation
- Displacement actual vs percentiles históricos
- Probabilidad de reversión/continuación
- Small wick warning

### 6. P1/P2 Weekly Analysis
- Detección de primer extremo semanal
- Validación de estructura (aceptación/mecha)
- P1 Flip Risk por tipo
- Timing P1→P2 y proyección

### 7. Síntesis de Decisión
- Semáforo multi-señal (TIME/DISTANCE/P1/BIAS)
- Score 0-4 ⭐
- Lectura narrativa final

## 🎓 Sistema Educativo

Cada sección tiene un botón **"📖 Explicar"** con explicaciones en lenguaje simple:
- ¿Qué son los pivots y cómo usarlos?
- ¿Cómo leer las proyecciones de precio?
- ¿Por qué importa la hora de formación?
- ¿Qué significa "small wick"?
- ¿Qué es P1/P2 y cómo difiere de TIME validation?
- ¿Cómo integrar todas las señales?

## ⚙️ Configuración

Edita las constantes en `CONFIG` al inicio del archivo:

```python
CONFIG = {
    "OUTPUT_DIR": "pivot_v45_output",
    "CACHE_DIR":  "pivot_v45_cache",
    "ASSETS":     ["BTC", "ETH"],  # Agregar más: ["BTC", "ETH", "SOL"]
}
```

## 📈 Interpretación de Señales

### Score de Estrellas
- **4/4 ⭐**: Alta confianza, todas las señales alineadas → OPERAR
- **3/4 ⭐**: Confianza moderada, mayoría alineada → Operar con cautela
- **2/4 ⭐**: Señales mixtas → Esperar confirmación
- **0-1/4 ⭐**: Conflicto de señales → NO OPERAR

### Alerta de Contexto
La alerta aparece automáticamente cuando detecta:
- **🟡 Amarillo**: Rebote/Pullback técnico (precaución)
- **🔴 Rojo**: Trampa sin confirmar (peligro)
- **🟢 Verde**: Alineación completa (señal clara)

## 🔧 Troubleshooting

**Error: "ccxt no instalado"**
```bash
pip install ccxt
```

**Error: "HTTPError 422" en Yahoo Finance**
- El bot usa CCXT como fuente principal
- Yahoo Finance solo como fallback

**Dashboard no se actualiza**
- El auto-refresh es cada 60 minutos
- Refrescar manualmente: F5 en navegador

**No puedo acceder desde celular**
- Verifica que estés en la misma red WiFi
- Usa la IP que muestra el bot (ej: 192.168.1.105:8080)
- Desactiva firewall temporalmente si es necesario

## 📝 Notas Importantes

- **No es consejo financiero**: Esta herramienta es para análisis educativo
- **Backtest antes de operar**: Valida el sistema con datos históricos
- **Risk management**: Siempre usa stop loss
- **Regla de oro**: No operar si score < 2 estrellas

## 🤝 Contribuir

Contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -m 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

MIT License - Ver archivo LICENSE para detalles

## 🙏 Agradecimientos

- Metodología ICT (Inner Circle Trader) para TIME/DISTANCE validation
- Conceptos SMC (Smart Money Concepts) para P1/P2 detection
- Comunidad de trading por feedback y testing

## 📬 Contacto

- Issues: [GitHub Issues](https://github.com/TU_USUARIO/crypto-pivot-analyzer/issues)
- Discussions: [GitHub Discussions](https://github.com/TU_USUARIO/crypto-pivot-analyzer/discussions)

---

**⚠️ DISCLAIMER**: Este software se proporciona "tal cual", sin garantías de ningún tipo. El trading de criptomonedas implica riesgo significativo de pérdida. Use bajo su propio riesgo.
