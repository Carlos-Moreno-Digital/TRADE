# Skill: Sistema Multi-Agente

## Cuando usar
Para entender, modificar o anadir agentes al sistema de TRADE.

## Arquitectura de agentes
El sistema tiene 7 agentes especializados + un orquestador central:
- Todos los agentes heredan de BaseAgent en agents/base.py
- Cada agente implementa: analyze(context: AnalysisContext) -> AgentSignal
- El orquestador en agents/orchestrator.py agrega las senales

## Como anadir un nuevo agente
1. Crear nuevo archivo en src/trade/agents/
2. Heredar de BaseAgent
3. Implementar analyze(context: AnalysisContext) -> AgentSignal
4. Registrar en agents/__init__.py
5. Anadir al pipeline del orquestador en orchestrator.py

## AgentSignal
Cada agente devuelve un AgentSignal con:
- direction: BUY, SELL o HOLD
- confidence: float de 0.0 a 1.0
- reasoning: string explicando la decision
- metadata: dict con datos adicionales (indicadores usados, etc.)

## Tipos de agentes disponibles
- Tecnico: basado en indicadores (RSI, MACD, Bollinger)
- Sentimiento: basado en noticias y LLM
- Macro: basado en datos macroeconomicos
- Volumen: analisis de flujo de volumen
- Patron: reconocimiento de patrones de velas

## Orquestacion
El orquestador agrega senales con pesos configurables en config/default.yaml.
Si el consenso es bajo, la decision es HOLD por defecto.
