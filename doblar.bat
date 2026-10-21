@echo off
chcp 65001 >nul
color 0B
echo ===================================================
echo     INICIANDO DOBLAJE AUTOMATICO CON IA (DOCKER)
echo ===================================================
echo.
echo Asegurate de tener Docker Desktop abierto.
echo Descargando imagenes e iniciando el sistema...
echo Esto puede tardar varios minutos la primera vez.
echo.

docker-compose up -d --build

if %ERRORLEVEL% neq 0 (
    echo.
    color 0C
    echo [ERROR] No se pudo iniciar el sistema.
    echo Asegurate de que Docker Desktop esta corriendo y tienes espacio en disco.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [EXITO] Sistema iniciado en segundo plano.
echo Esperando unos segundos para que la interfaz web este lista...
timeout /t 5 /nobreak >nul

echo Abriendo navegador en http://localhost:8080 ...
start http://localhost:8080

echo.
echo ========================================================
echo ¡Todo listo! El sistema se esta ejecutando en tu navegador.
echo Puedes cerrar esta ventana negra de forma segura, 
echo el sistema seguira funcionando silenciosamente.
echo ========================================================
timeout /t 5 >nul
