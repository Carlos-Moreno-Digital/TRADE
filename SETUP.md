# Guia de Instalacion - TRADE Bot

## Requisitos

- **Python 3.11 o superior** (descarga: https://www.python.org/downloads/)
  - Al instalar, marca la casilla **"Add Python to PATH"**
- **Git** (descarga: https://git-scm.com/downloads)
- **Conexion a internet** (para descargar datos de mercado)

## Paso 1: Descargar el codigo

Abre una terminal (CMD en Windows, Terminal en Mac) y ejecuta:

```bash
git clone https://github.com/Carlos-Moreno-Digital/TRADE.git
cd TRADE
```

## Paso 2: Crear entorno virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac / Linux
python3 -m venv .venv
source .venv/bin/activate
```

## Paso 3: Instalar dependencias

```bash
pip install -e ".[dev]"
```

Si da error con `ta-lib`, en Windows necesitas:
1. Descarga el instalador de https://github.com/cgohlke/talib-build/releases
2. Busca el archivo `.whl` que coincida con tu version de Python (cp311 = Python 3.11)
3. Instala con: `pip install TA_Lib-0.6.x-cp311-cp311-win_amd64.whl`

## Paso 4: Configurar credenciales

Copia el archivo de ejemplo:
```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Edita `.env` con tus datos de TradeLocker:
```
TRADELOCKER_USERNAME=tu_email@ejemplo.com
TRADELOCKER_PASSWORD=tu_password_tradelocker
TRADELOCKER_SERVER=FunderPro
TRADELOCKER_ENVIRONMENT=https://demo.tradelocker.com
```

## Paso 5: Verificar instalacion

```bash
python -m trade.main --help
```

Deberia mostrar todas las opciones disponibles.

## Paso 6: Ejecutar modo demo

El modo demo analiza el mercado pero **NO ejecuta trades reales**:

```bash
# Analizar EUR/USD, GBP/USD y Oro cada 15 minutos
python -m trade.main --demo

# Analizar un par especifico
python -m trade.main --demo --symbol EURUSD=X

# Cambiar intervalo a 5 minutos
python -m trade.main --demo --interval 5
```

Para parar: pulsa **Ctrl+C**

## Paso 7: Ver reportes

Despues de que el bot haya estado corriendo un rato:

```bash
# Ver reporte de los ultimos 7 dias
python -m trade.main --report

# Ver reporte de los ultimos 30 dias
python -m trade.main --report --report-days 30
```

## Comandos utiles

| Comando | Que hace |
|---------|----------|
| `python -m trade.main --demo` | Modo demo (observacion) |
| `python -m trade.main --report` | Ver reportes |
| `python -m trade.main --symbol EURUSD=X` | Analisis rapido EUR/USD |
| `python -m trade.main --watchlist` | Analizar todos los pares |
| `pytest tests/` | Ejecutar tests |

## Problemas comunes

### "No module named trade"
Asegurate de haber activado el entorno virtual:
- Windows: `.venv\Scripts\activate`
- Mac/Linux: `source .venv/bin/activate`

### "ta-lib not found"
Sigue las instrucciones de instalacion de TA-Lib arriba.

### "Connection refused" en TradeLocker
Verifica que tus credenciales en `.env` son correctas y que tienes
una cuenta activa en TradeLocker/FunderPro.

### El bot no hace nada
El bot solo opera durante las "kill zones" (horas de alta probabilidad):
- **London Open**: 07:00-10:00 UTC (09:00-12:00 hora espanola)
- **New York Open**: 12:00-15:00 UTC (14:00-17:00 hora espanola)
- **London/NY Overlap**: 12:00-16:00 UTC (14:00-18:00 hora espanola)

Fuera de estas horas, el bot analiza pero probablemente dira HOLD.
