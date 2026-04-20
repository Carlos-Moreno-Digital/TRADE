# Comando: Safety Check de Trading

Antes de cualquier modificacion al sistema de trading, verificar:

## Modo de operacion
- [ ] El modo por defecto es PAPER TRADING (nunca cambiar sin confirmacion explicita)
- [ ] config/default.yaml tiene mode: paper
- [ ] No hay conexiones a brokers reales activas

## Circuit breakers
- [ ] CircuitBreaker esta instanciado en el orquestador
- [ ] Los limites estan configurados: max_drawdown 10%, max_daily_loss 3%, max_trade_loss 2%
- [ ] El circuit breaker loguea cuando se activa

## Logging
- [ ] Todas las decisiones de agentes se loguean con razonamiento
- [ ] Las ordenes ejecutadas se loguean con timestamp, precio y size

## Configuracion de riesgo
- [ ] Stop loss definido para cada estrategia
- [ ] Position sizing dentro de limites configurados
- [ ] No se han deshabilitado validaciones de riesgo

## Tests
- [ ] pytest tests/ pasa sin errores
- [ ] Los tests de risk management estan pasando
