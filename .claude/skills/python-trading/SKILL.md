# Skill: Python para Trading

## Cuando usar
Para cualquier desarrollo en el proyecto TRADE: agentes, indicadores, datos, estrategias.

## Reglas de estilo del proyecto
- ruff check src/ para linting (no flake8)
- ruff format src/ para formateo (no black)
- pytest tests/ para tests
- Type hints en todas las funciones publicas
- Docstrings en clases y metodos publicos

## Librerias clave del proyecto
- yfinance: datos de mercado (OHLCV, info de empresa)
- pandas-ta: indicadores tecnicos (RSI, MACD, Bollinger Bands)
- pandas: manipulacion de dataframes OHLCV
- numpy: calculos numericos
- OpenBB: datos alternativos y financieros avanzados

## Convencion de datos OHLCV
- Dataframe siempre con DatetimeIndex
- Columnas: Open, High, Low, Close, Volume (mayusculas)
- Timezone: UTC siempre (convertir a local solo para display)

## Tipos de datos
- Precios: float (no Decimal, overhead innecesario)
- Cantidades: int cuando son enteros, float cuando fraccionados
- Fechas: pd.Timestamp o datetime con timezone UTC

## Comandos del proyecto
pip install -e .[dev]                              # Instalar
python -m trade.main --symbol AAPL --mode paper      # Ejecutar (siempre paper por defecto)
pytest tests/                                        # Tests
