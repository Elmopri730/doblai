# Guía de Instalación y Uso (Para Principiantes)

¡Bienvenido a tu **Sistema Automatizado de Doblaje de Vídeo**! 

Esta guía está diseñada para que puedas poner en marcha el sistema en tu ordenador de la forma más sencilla posible, **sin tocar una sola línea de código**. Todo funciona haciendo doble clic y usando tu navegador.

---

## 1. Requisitos Previos (Solo necesitas hacer esto 1 vez)

Para que el sistema funcione, necesitamos un "motor" que haga correr la Inteligencia Artificial por debajo. Ese motor se llama **Docker Desktop**.

1. **Descarga Docker Desktop** haciendo clic en este enlace: [Descargar Docker Desktop para Windows](https://docs.docker.com/desktop/install/windows-install/).
2. **Instálalo** haciendo doble clic en el archivo que acabas de descargar. Dale a "Siguiente" o "Aceptar" a todo (deja las opciones por defecto).
3. **Abre Docker Desktop** desde tu menú de inicio de Windows. 
4. *Nota:* La primera vez que lo abras, puede que te pida iniciar sesión (puedes saltarte este paso si quieres) o que tarde un minuto en arrancar el motor (verás que la barrita inferior se pone en verde). Una vez en verde, ¡ya tienes el motor encendido!

---

## 2. Cómo Arrancar y Usar el Sistema (Día a Día)

Una vez tienes Docker Desktop abierto, arrancar la aplicación de doblaje es tan simple como abrir un programa normal:

1. Ve a la carpeta donde tienes guardado este proyecto.
2. Haz **doble clic** en el archivo llamado `doblar.bat` (puede que solo veas `doblar` si tienes las extensiones ocultas; tiene el icono de una rueda dentada o ventana de consola).
3. Verás que se abre una ventana negra durante unos segundos. ¡No te asustes! Está encendiendo las inteligencias artificiales por debajo.
4. **Automáticamente se abrirá tu navegador web** (Chrome, Edge, etc.) en la dirección `http://localhost:8080`.
5. ¡Listo! Ya estás en la aplicación. Solo tienes que **arrastrar tu vídeo** a la pantalla y pulsar "Doblar".

> [!TIP]
> **Puedes cerrar la ventana negra** una vez se haya abierto el navegador. El sistema seguirá funcionando silenciosamente en segundo plano.

---

## 3. Cómo Detener el Sistema

Si has terminado de doblar vídeos y quieres liberar la memoria de tu ordenador, debes apagar el sistema:

1. Ve a la carpeta del proyecto.
2. Haz **doble clic** en el archivo llamado `detener.bat`.
3. Espera unos segundos a que la ventana negra se cierre sola. ¡Ya está apagado!

*(También puedes simplemente cerrar el programa "Docker Desktop" desde la barra de tareas de Windows, lo cual apagará el motor por completo).*

---

## 4. Si algo no funciona (Solución de problemas)

Aquí tienes los problemas más comunes y su solución en menos de un minuto:

> [!WARNING]
> **Problema:** Hago doble clic en `doblar.bat` y me dice "Docker no está instalado o no está ejecutándose".
> **Solución:** Significa que el motor está apagado. Abre la aplicación "Docker Desktop" desde tu menú de Inicio de Windows, espera a que cargue y vuelva a estar en verde, y vuelve a probar.

> [!WARNING]
> **Problema:** El navegador se abre pero la página dice "No se puede acceder a este sitio web" (ERR_CONNECTION_REFUSED).
> **Solución:** A veces el sistema tarda unos segundos extra en arrancar. Espera 10 segundos y pulsa la tecla **F5** (o el botón de actualizar) en tu navegador.

> [!WARNING]
> **Problema:** Sale un error sobre que el puerto `8080` ya está ocupado.
> **Solución:** Tienes otra aplicación en tu ordenador usando ese "espacio". Para solucionarlo temporalmente, apaga otros programas web que tengas corriendo en segundo plano, o contacta con el administrador para cambiar el puerto en la configuración del proyecto.

> [!IMPORTANT]
> **Problema:** El doblaje de voz funciona muy lento.
> **Solución:** Este sistema procesa vídeo e inteligencia artificial, lo cual requiere potencia. Si no tienes una tarjeta gráfica (GPU) dedicada, el sistema usará tu procesador normal (CPU), que es más lento pero igual de efectivo. Solo requiere un poco más de paciencia.
