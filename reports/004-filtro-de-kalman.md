# 004 — Filtro de Kalman: velocidade constante, Σd de aceleração aleatória, validação contra o OpenCV

**Data:** 21/11/2026 · **Código:** `src/kalman.py`, `src/synthetic.py` · **Testes:** `tests/test_kalman_vs_opencv.py`

## Contexto

O enunciado exige predição e correção programadas pela dupla, com x, D, M, Σd e Σm explícitos e justificados, estado mínimo [px, py, vx, vy]ᵀ, medição só da posição, e o `cv2.KalmanFilter` usado **apenas para validar** (os resultados precisam coincidir).

## Decisão

**Implementação** em NumPy, com a notação da aula (classe `KalmanFilter`):

- **Predição:** x⁻ = D x ;  P⁻ = D P Dᵀ + Σd
- **Inovação:** y = z − M x⁻ ;  S = M P⁻ Mᵀ + Σm
- **Ganho:** K = P⁻ Mᵀ S⁻¹
- **Correção:** x = x⁻ + K y ;  P = (I − K M) P⁻
- **Sem medição:** a predição vira a estimativa (o mesmo comportamento do OpenCV).
- `correct()` aceita um **Σm por quadro**, já prevendo o filtro F2 em metros (Σm propagado pelo jacobiano da homografia).

**Modelo de velocidade constante** (`constant_velocity(dt, sigma_a, sigma_m)`), com Δt = 1/fps:

- D = [[1,0,Δt,0],[0,1,0,Δt],[0,0,1,0],[0,0,0,1]] ;  M = [[1,0,0,0],[0,1,0,0]]
- **Σd** pelo modelo de **aceleração aleatória** (ruído branco na aceleração, intensidade σa²), por eixo: σa² · [[Δt⁴/4, Δt³/2], [Δt³/2, Δt²]]
- **Σm** = σm² · I

**Justificativa:** a bola rola com velocidade aproximadamente constante entre dois toques; chutes e atrito entram como "acelerações não modeladas". σa é o botão do compromisso **elefante × peixinho-dourado**: com σa pequeno o filtro é teimoso (suave, mas atrasa), com σa grande é nervoso (acompanha rápido, mas treme).

**Inicialização:** posição = primeira medição, velocidade zero, com incerteza σv0 = 300 px/s.

**Simulador sintético** (`src/synthetic.py`):

- bola com toques aleatórios e atrito opcional;
- ruído gaussiano na medição;
- oclusões e falhas aleatórias do detector;
- candidatos falsos aleatórios (Poisson) e parados ("folhas").

Ele permitiu desenvolver e testar o filtro antes de existir vídeo.

## Alternativas

- **Usar `cv2.KalmanFilter` no pipeline:** proibido pelo enunciado; ele aparece só no teste.
- **Forma de Joseph para P** ((I−KM)P⁻(I−KM)ᵀ + KΣmKᵀ): numericamente mais robusta, mas a forma simples é a da aula e a do OpenCV, o que permite a comparação exata. Fica como possível melhoria.
- **Modelo de aceleração constante:** extensão possível. Para bola rasteira, a velocidade constante com σa adequado basta.

## Evidência

- **Validação contra o OpenCV:** com as mesmas matrizes e a mesma sequência (250 quadros, oclusões em 60–85 e 150–160, 10% de falhas aleatórias), x⁻, P⁻, x e P coincidem com `rtol = atol = 1e-9`, para **σa = 5, 200 e 3000**, inclusive nos quadros só com predição.
- **Redução de ruído** (sintético, σm = 2 px, movimento suave, σa = 100): erro médio da estimativa < 0,8 × erro médio da medição.
  - Numa varredura: σa = 60 → 1,55 px contra 2,47 px da medição; σa = 200 → 1,43 px; σa = 5 → 8,4 px (teimoso demais, atrasa nos toques); σa = 1000 → 1,83 px.
  - Esse padrão em "U" é exatamente o compromisso elefante × peixinho-dourado que o experimento E2 vai mostrar nos vídeos.
- **Matrizes:** Σd simétrica e semidefinida positiva; os termos conferem com as fórmulas.

Comando: `python -m pytest -q tests/test_kalman_vs_opencv.py`.

## Consequências

- **Requisito de 30% da nota** (corretude do filtro): validado por teste automatizado.
- **σm = 1,5 px era provisório:** medido em 22/11 nos vídeos de teste; ver a Revisão abaixo (o detector tem 0,1–0,5 px; 1,5 px fica como ruído efetivo).
- **σa nos vídeos** foi ajustado pela trilha ([005](005-gating-e-trilha.md)): o valor intermediário atual é 4000 px/s². Os chutes mudam a velocidade da bola em um ou dois quadros, por isso os valores são altos em px/s².

## Revisão (22/11): σm medido nos vídeos de teste (T20)

**Código:** `experiments/sigma_m.py` (testes em `tests/test_sigma_m.py`). Usa o cache de candidatos de `experiments/replay_tracker.py` ([013](013-limites-fisicos-no-f2.md)); o candidato-bola é o mais próximo da marcação (interpolada entre quadros marcados), a ≤ 10 px.

**Medidas** (px a 640, no quadro de referência, que é onde o filtro F1 recebe a medição):

| | Lateral | Centro |
|---|---|---|
| Bola parada: desvio-padrão x / y | **0,15 / 0,08 px** (quadros 0–100, n = 101) | 0,84 / 0,31 px (quadros 0–6, n = 7: pouco, não conclusivo) |
| Bola parada, no quadro atual (sem estabilizar) | 0,72 / 0,55 px | 0,74 / 0,51 px |
| Em movimento: resíduo de parábola local (7 quadros), MAD x / y | **0,29 / 0,21 px** (n = 891) | **0,39 / 0,34 px** (n = 1346) |
| Em movimento: desvio-padrão do resíduo (inclui chutes) x / y | 0,64 / 0,39 px | 0,54 / 0,42 px |

Comandos:

```bash
python -m experiments.sigma_m videos/campo_lateral.MOV --static 0 100
python -m experiments.sigma_m videos/campo_centro.MOV --static 0 6
```

**Leitura:**

- O ruído do detector + estabilização é de **0,1–0,2 px com a bola parada** e **0,3–0,5 px com a bola rolando** (borrão de movimento). Sem a estabilização, o balanço da câmera somaria ~0,7 px.
- O σm provisório de 1,5 px é 3–5 vezes maior que o medido.

**Efeito de usar o σm medido** (σa = 4000 px/s² ou 150 m/s² fixos; F2 com o limite de 20 m/s do [013](013-limites-fisicos-no-f2.md); `python -m experiments.replay_tracker videos/<v>.MOV [--metric] --sigma-m <s>`):

| Na bola (teste) | σm = 1,5 | 0,5 | 0,3 |
|---|---|---|---|
| Lateral F1 | **63,7%** (K 0,90) | 59,5% (K 0,97) | 57,0% (K 0,99) |
| Lateral F2 | **66,2%** | 65,4% | 65,4% |
| Centro F1 | **66,3%** | 62,6% | 61,8% |
| Centro F2 | **68,0%** | 67,1% | 65,7% |

**Decisão: o padrão continua σm = 1,5 px**, agora como **ruído efetivo**, e não como o ruído do detector.

- Com σa fixo, um σm menor leva o K para 0,97–0,99 (o filtro copia ainda mais o detector) e encolhe a região de busca (S = M P⁻ Mᵀ + Σm). Menos medições são aceitas: lateral F1 com 305 em vez de 321 aceitas com a bola visível, e 6 trilhas a mais.
- O 1,5 px cobre, além do ruído do detector, o que o modelo de velocidade constante não explica entre quadros (atrito, quiques, centroide que muda quando o blob é parcial ou borrado). O K em regime só depende da razão σa/σm. Usar σm = 0,5 com σa = 1333 dá o mesmo K = 0,90, mas uma região de busca 3 vezes menor em px.
- **Consequência para o E2 (Tarefa 3):** a varredura de σa deve ser feita com σm = 1,5 fixo, e o relatório deve dizer que o σm do filtro é maior que o ruído medido do detector, e por quê. A alternativa de ajustar os dois juntos (σm medido e σa menor) fica registrada aqui como testada e pior.
