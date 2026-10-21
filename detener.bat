@echo off
chcp 65001 >nul
color 0E
echo ===================================================
echo     DETENIENDO EL SISTEMA DE DOBLAJE...
echo ===================================================
echo.
echo Cerrando contenedores y liberando memoria VRAM/RAM...

docker-compose down

echo.
echo [OK] El sistema se ha detenido correctamente.
echo Ya puedes cerrar esta ventana.
pause
