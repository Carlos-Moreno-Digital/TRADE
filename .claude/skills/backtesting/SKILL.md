# Skill: Backtesting

## Cuando usar
Para diseno, implementacion y analisis de backtests en TRADE.

## Principios de backtesting riguroso
1. Sin look-ahead bias: los indicadores solo usan datos disponibles en ese momento
2. Walk-forward validation: no usar todo el historico para optimizar
3. Comisiones y slippage: siempre incluirlos (el backtest sin ellos miente)
4. Out-of-sample: guardar al menos el 20% del historico para validacion final

## Metricas obligatorias en cada backtest
- Sharpe Ratio: objetivo > 1.0 (> 1.5 es bueno, > 2.0 es excelente)
- Max Drawdown: limite configurado en config (default: 10%)
- Win Rate: porcentaje de operaciones ganadoras
- Profit Factor: suma ganancias / suma perdidas (objetivo > 1.5)
- Total Return: retorno total del periodo
- CAGR: tasa de retorno anual compuesta

## Estructura de backtest en TRADE
- Los backtests viven en src/trade/analysis/ o en archivos backtest_*.py
- Usar BacktestContext para pasar datos y configuracion
- Loguear cada operacion con entrada, salida, razon y P&L

## Advertencias criticas
- Un buen backtest NO garantiza rendimiento futuro
- Overfitting: si optimizas demasiado en historico, fallara en produccion
- Regimen de mercado: una estrategia puede funcionar en tendencia y fallar en lateral
- Siempre validar con datos de periodo diferente al de optimizacion
