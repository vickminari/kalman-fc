# 010 — O que o Kalman está fazendo no vídeo e de onde vêm as "trajetórias loucas"

**Data:** 22/11/2026

**Base:** vídeo lateral, F1 com score combinado (`outputs/lateral_F1/track.csv`), 810 quadros marcados. Conceitos da aula 08 (Tracking, Prof. Kelson).

## Contexto

Assistindo ao vídeo, a dupla perguntou:

- em que o filtro está contribuindo, já que a estimativa parece "grudada" no detector, sempre em volta da bola;
- por que aparecem trajetórias estranhas "do nada".

## Diagnóstico (números)

**1. Com σa = 4000 px/s², o filtro quase copia o detector.**

Ganho da posição em regime (σm = 1,5 px, 30 fps):

| σa (px/s²) | 100 | 800 | 4000 (atual) | 15000 |
|---|---|---|---|---|
| K da posição | 0,32 | 0,66 | **0,90** | 0,98 |

- **Distância estimativa → medição** nos quadros com medição: mediana de 0,06 px.
- **Tremor** (segunda diferença): medição 0,61 px, estimativa 0,55 px. Quase não há suavização.

É o lado "peixinho-dourado" do compromisso da aula: σa foi aumentado para não perder a bola nos chutes, e com isso o filtro passou a confiar quase só na medição. Na regra da aula: "se o ruído de medição for pequeno [em relação ao do processo], confiamos principalmente na medição".

**2. O que o Kalman acrescenta hoje é a associação e a predição, não a suavização.**

- **Escolher a bola entre os candidatos** (gating): com a bola visível, 306 de 321 medições aceitas são a bola (95%). O melhor candidato do detector, sem trilha, é a bola em 38% dos quadros.
- **Atravessar falhas do detector:** 26,5% dos quadros ficam "no escuro" e são cobertos só pela predição.
- **Velocidade** (km/h no minimapa).

**3. As "trajetórias loucas" vêm, em ordem:**

- **Predição no escuro andando com uma velocidade ruidosa.** De 118 saltos > 20 px entre quadros consecutivos, **62** acontecem no escuro. Lá a predição anda 3 px/quadro na mediana, mas 16 px/quadro no p90 (~480 px/s). Com o filtro nervoso, a velocidade é praticamente a diferença entre duas medições ruidosas, e qualquer associação errada ou chute vira uma velocidade grande, que a predição segue em linha reta por até 20 quadros.
- **Retomada longe da predição:** 40 saltos. A medição volta na borda da região de busca (até 60 px), seja um chute de verdade ou outro objeto.
- **Trilhas novas:** 12 saltos (59 trilhas no vídeo). Em 6 delas o quadro de início estava marcado, e só 3 nasceram na bola.
- **No F2:** no fundo do campo, 1 px vale muitos metros, e o mesmo erro vira um salto maior no minimapa. Bola no ar ou encostada na parede sai do plano da homografia.

**4. O gargalo continua na entrada.** Com a bola visível, em 189 de 510 quadros não há medição aceita: o detector não acha a bola ou o score combinado a descarta. Pés e mãos aparecem porque, sob os refletores, a pele clara tem saturação baixa e brilho alto, parecida com o branco da bola.

## Decisão: próximos passos (ordem proposta em 22/11)

**A. Software, sem gravar nada:**

1. **Visualização honesta da predição:**
   - rastro só dos quadros com medição;
   - predição no escuro tracejada e esmaecida;
   - parar de desenhar a predição depois de ~10 quadros no escuro.
2. **Limites físicos no F2,** em metros, onde fazem sentido (em px foram descartados em [005](005-gating-e-trilha.md)):
   - velocidade máxima de ~30 m/s;
   - `max_dark` menor (~10–15 quadros);
   - medir de novo.
3. **E2 como ferramenta, não só como relatório:** varrer σa (por exemplo 300, 1000, 4000 px/s² e os equivalentes em m/s²) e mostrar o compromisso. Um σa intermediário (K ~ 0,6–0,7) suaviza mais, mas perde a bola nos chutes.
   - **Melhoria possível (mais trabalho):** ruído de processo adaptativo, ou seja, aumentar Σd quando a inovação for grande (detecção de manobra). Assim, suaviza na rolagem e acompanha no chute.
4. **Falsos positivos de pele:** medir HSV dos candidatos falsos em pés e mãos e, se separar, acrescentar ao score um atributo de cor de pele (matiz 5–25 com saturação média).
5. **σm medido** com a bola parada (início do lateral).

**B. Gravação "limpa" (opcional, recomendada se for fácil, 1–2 min):**

- câmera no tripé (dispensa a estabilização e permite a subtração de fundo MOG2);
- **bola colorida** (laranja ou amarela; a bola branca concorre com a parede branca e com a pele sob luz forte);
- gramado limpo, bola sempre rasteira e quase sempre no quadro;
- 1–2 pessoas;
- algumas oclusões propositais de 10–30 quadros;
- mesmo enquadramento do lateral (traves visíveis) para reaproveitar a calibração.

**Uso:** os vídeos de teste continuam a base ([008](008-videos-de-teste-como-base.md)). O clipe limpo serviria para E1–E3 com resultados "de livro" e para o vídeo de demonstração. Os de teste ficam como o caso difícil, para discussão.

**C. Medidas extras:** não são necessárias para a limpeza do resultado. Só a distância linha do gol → muro do fundo, para tirar os priors da calibração do centro.

## Consequências

- **No relatório,** a discussão do E2 tem um exemplo concreto: com σa alto, K = 0,9 e quase nenhuma suavização, mas a trilha não se perde nos chutes.
- **"Trajetória louca"** não é defeito do filtro de Kalman em si: é a predição sem medição seguindo uma velocidade mal estimada (entrada ruidosa e associações erradas). Os passos A1–A3 atacam isso diretamente.
