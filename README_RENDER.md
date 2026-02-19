# 🚀 Deploy a Render.com - Guía Paso a Paso

Esta guía te llevará desde cero hasta tener tu dashboard online y accesible 24/7 desde cualquier parte del mundo.

---

## 📋 PREREQUISITOS

- Cuenta de GitHub (gratis)
- Cuenta de Render.com (gratis)
- Los archivos del bot descargados

---

## PASO 1: SUBIR A GITHUB

### 1.1 Crear repositorio en GitHub

1. Ve a https://github.com/new
2. Nombre del repositorio: `crypto-pivot-analyzer`
3. Descripción: `Sistema de análisis técnico crypto con TIME/DISTANCE validation`
4. **Público** (para usar el free tier de Render)
5. ❌ NO marques "Add README" ni "Add .gitignore"
6. Click "Create repository"

### 1.2 Subir archivos

**Opción A - Interfaz web** (MÁS FÁCIL):

1. En tu repositorio, click "uploading an existing file"
2. Arrastra TODOS estos archivos:
   - `app.py`
   - `crypto_pivot_v4_5_hybrid.py`
   - `requirements.txt`
   - `.gitignore`
   - `LICENSE`
   - `README.md`
   - `render.yaml`
3. Commit message: `Initial commit`
4. Click "Commit changes"

**Opción B - Git CLI**:

```bash
cd tu-carpeta-local
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/TU_USUARIO/crypto-pivot-analyzer.git
git branch -M main
git push -u origin main
```

---

## PASO 2: DEPLOY EN RENDER

### 2.1 Crear cuenta en Render

1. Ve a https://render.com
2. Click "Get Started"
3. Regístrate con GitHub (click "Sign up with GitHub")
4. Autoriza Render a acceder a tu GitHub

### 2.2 Crear Web Service

1. En el dashboard de Render, click "New +"
2. Click "Web Service"
3. Click "Connect a repository"
4. Si no ves tu repositorio:
   - Click "Configure account" (arriba a la derecha)
   - Selecciona tu repositorio `crypto-pivot-analyzer`
   - Click "Save"
   - Vuelve a "New +" → "Web Service"
5. Selecciona `crypto-pivot-analyzer`

### 2.3 Configurar el servicio

**Configuración básica**:
- **Name**: `crypto-pivot-analyzer` (o el nombre que prefieras)
- **Region**: US West (Oregon) - o el más cercano a ti
- **Branch**: `main`
- **Root Directory**: dejar vacío
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py`

**Plan**:
- Selecciona **"Free"** ($0/mes)
- ⚠️ Nota: Free tier se duerme después de 15 min sin uso, tarda ~30s en despertar

**Advanced** (opcional):
- **Auto-Deploy**: ✅ Yes (auto-deploy cuando hagas push a GitHub)
- **Health Check Path**: `/dashboard.html`

### 2.4 Deploy

1. Click "Create Web Service"
2. Espera 3-5 minutos mientras Render:
   - Clona tu repositorio
   - Instala dependencias
   - Inicia el servidor
3. Verás logs en tiempo real
4. Cuando veas `✓ Servidor iniciado en puerto 10000` → ¡LISTO!

---

## PASO 3: ACCEDER A TU DASHBOARD

Tu dashboard estará disponible en:

```
https://tu-nombre-app.onrender.com/dashboard.html
```

**Ejemplo**:
```
https://crypto-pivot-analyzer-xyz123.onrender.com/dashboard.html
```

### URL personalizada (opcional)

Si quieres cambiar la URL:
1. En Render, ve a tu servicio
2. Click "Settings"
3. En "Name", cambia el nombre
4. La URL cambiará automáticamente

---

## PASO 4: CONFIGURAR AUTO-UPDATE

El dashboard se regenera automáticamente cada 60 minutos. Si quieres cambiar esto:

1. Edita `app.py` en GitHub:
```python
# Línea ~XX (busca "time.sleep(3600)")
time.sleep(3600)  # 3600 segundos = 60 minutos

# Cámbialo a lo que quieras:
time.sleep(1800)  # 30 minutos
time.sleep(900)   # 15 minutos
```

2. Commit el cambio
3. Render auto-deploya la nueva versión

---

## 🎨 PERSONALIZACIÓN

### Agregar más activos

Edita `app.py`:
```python
CONFIG = {
    "ASSETS": ["BTC", "ETH", "SOL", "AVAX"],  # Agregar más
}
```

### Cambiar frecuencia de actualización

```python
time.sleep(1800)  # 30 minutos en lugar de 60
```

---

## 🔧 TROUBLESHOOTING

### "Application failed to respond"

**Causa**: El bot está descargando datos por primera vez
**Solución**: Espera 5 minutos y recarga

### "Service unavailable"

**Causa**: Free tier se durmió por inactividad
**Solución**: Espera 30 segundos, se despertará automáticamente

### "Build failed"

**Causa**: Error en requirements.txt o código
**Solución**: 
1. Revisa los logs en Render
2. Verifica que todos los archivos estén subidos a GitHub
3. Re-deploy desde Render

### Dashboard no se actualiza

**Causa**: El thread de regeneración falló
**Solución**: 
1. Ve a Render → tu servicio → "Manual Deploy" → "Clear build cache & deploy"
2. O haz un push vacío a GitHub: `git commit --allow-empty -m "Trigger redeploy" && git push`

### Datos históricos toman mucho tiempo

**Causa**: Descargando 12 meses de datos 4H por primera vez
**Solución**: Normal, tarda 2-3 minutos en el primer deploy. Después usa cache.

---

## 📊 MONITOREO

### Ver logs en tiempo real

1. En Render, ve a tu servicio
2. Click en "Logs" (sidebar izquierdo)
3. Verás cada actualización del dashboard

### Métricas

En "Metrics" puedes ver:
- Requests por minuto
- Tiempo de respuesta
- CPU/Memory usage

---

## 💰 COSTO

**Free Tier**:
- ✅ 750 horas/mes gratis
- ✅ Auto-sleep después de 15 min sin tráfico
- ✅ SSL gratis (HTTPS)
- ❌ Se duerme con inactividad (~30s para despertar)

**Paid Tier** ($7/mes):
- ✅ Siempre activo (no se duerme)
- ✅ Mejor performance
- ✅ Más recursos

---

## 🔐 SEGURIDAD

### Agregar autenticación (opcional)

Si quieres proteger tu dashboard con usuario/contraseña:

1. Edita `app.py` y agrega:
```python
class AuthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        auth = self.headers.get('Authorization')
        if auth != 'Basic dXNlcjpwYXNzd29yZA==':  # user:password en base64
            self.send_response(401)
            self.send_header('WWW-Authenticate', 'Basic realm="Dashboard"')
            self.end_headers()
            return
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
```

2. Usa `AuthHandler` en lugar de `DashboardHandler`

---

## 🚀 SIGUIENTES PASOS

Una vez funcionando:

1. **Comparte la URL** con quien quieras
2. **Guarda la URL** en tus favoritos
3. **Agrega a homescreen** en tu celular
4. **Monitorea diariamente** antes de operar

---

## ✅ CHECKLIST FINAL

- [ ] Repositorio creado en GitHub
- [ ] Todos los archivos subidos
- [ ] Servicio creado en Render
- [ ] Deploy exitoso (logs muestran "Servidor iniciado")
- [ ] Dashboard accesible desde la URL de Render
- [ ] Auto-update funcionando (verificar después de 60 min)
- [ ] URL guardada en favoritos

---

## 📬 AYUDA

Si algo no funciona:
1. Revisa los logs en Render
2. Verifica que todos los archivos estén en GitHub
3. Abre un Issue en tu repositorio de GitHub
4. Describe el error específico que ves

---

**🎉 ¡FELICITACIONES!** Tu dashboard está online y accesible desde cualquier parte del mundo 24/7.
