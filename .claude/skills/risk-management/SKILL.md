# Skill: Gestion de Riesgo

## Cuando usar
Para cualquier logica relacionada con riesgo, position sizing, circuit breakers y limites en TRADE.

## REGLAS ABSOLUTAS (no modificar sin confirmacion explicita)
- PAPER TRADING es el modo por defecto. Nunca cambiar a live sin confirmacion explicita del usuario.
- Circuit breakers son OBLIGATORIOS. Nunca deshabilitarlos.
- Todas las decisiones de agentes deben ser logueadas.

## Limites configurados (config/default.yaml)
- Max drawdown: 10% (si se supera, detener trading)
- Max perdida diaria: 3% (si se supera, detener hasta el dia siguiente)
- Max perdida por operacion: 2% del capital
- Max posicion en un solo activo: configurado por estrategia

## Position sizing (Kelly Criterion ajustado)
- No usar Kelly completo (demasiado agresivo)
- Usar Kelly / 4 como maximo por operacion
- Reducir size si la volatilidad del activo es alta

## Circuit breakers
Son la red de seguridad del sistema. Verificar que estan activos antes de cualquier test:
- Verificar en src/trade/risk/ que CircuitBreaker esta instanciado en el orquestador
- Los circuit breakers deben loguear cuando se activan

## Stop Loss
- Siempre definir stop loss antes de entrar en una operacion
- Stop basado en ATR es mas robusto que stop porcentual fijo
- Nunca mover el stop en contra de la posicion (ampliar la perdida)

## Alertas de riesgo
- Si el drawdown supera el 5%, loguear WARNING
- Si el drawdown supera el 8%, loguear ERROR
- Si el drawdown supera el 10%, activar circuit breaker CRITICO
