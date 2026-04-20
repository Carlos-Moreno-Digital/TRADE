# Skill: Analisis de Datos Financieros

## Cuando usar
Para descargar, limpiar, analizar y visualizar datos de mercado en TRADE.

## Fuentes de datos
- yfinance: datos OHLCV historicos y en tiempo real (gratis)
- OpenBB: datos financieros avanzados, fundamentales, alternativas
- Dukascopy: datos de forex de alta frecuencia (5min, 1min)

## Descargar datos con yfinance
Usar yf.download con auto_adjust=True para ajustar splits y dividendos.
Especificar siempre el intervalo (1d, 1h, 5m) y el periodo o fechas concretas.

## Indicadores tecnicos con pandas-ta
Anadir indicadores directamente al DataFrame con df.ta.indicador().
Los indicadores mas usados en el proyecto: RSI, MACD, Bollinger Bands, ATR, EMA.

## Limpieza de datos
- Eliminar NaN al inicio del DataFrame (causados por calculo de indicadores)
- Verificar que el DatetimeIndex no tiene huecos inesperados
- Detectar y manejar valores outliers (precios 0 o negativos)

## Visualizacion rapida
Usar matplotlib o plotly para visualizar series temporales de precios e indicadores.
Guardar graficos en data/charts/ con nombre descriptivo y fecha.

## Periodos de datos recomendados
- Desarrollo/test rapido: 3 meses
- Backtest inicial: 2 anos
- Backtest robusto: 5-10 anos (incluye diferentes regimenes de mercado)
- Walk-forward: dividir en ventanas de 6 meses
