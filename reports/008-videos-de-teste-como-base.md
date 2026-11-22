# 008 — Os vídeos de teste viram a base do trabalho; a gravação definitiva fica adiada

**Data:** 22/11/2026

## Contexto

O escopo previa uma "gravação definitiva" com cenas específicas (V0–V6: xadrez, campo com marcos medidos, bola parada, bola rolada entre cones, passes, oclusão proposital, jogo). A dupla perguntou se isso é mesmo necessário. Nenhuma cena era para treinamento: cada uma servia para **medir** algo (intrínsecos, homografia, σm, velocidade de referência) ou para cumprir o experimento obrigatório (iii) de oclusão.

Ao mesmo tempo, os dois vídeos de teste (lateral e centro) já estão **inteiramente marcados** (810 + 727 quadros) e o pipeline roda neles.

## Decisão (da dupla, 22/11)

1. **Fazer o máximo com os vídeos de teste.** A gravação definitiva fica adiada e só acontece se algo realmente exigir. Até lá, o código fica pronto para recebê-la sem mudanças.
2. **Oclusão (iii):** as **oclusões naturais** dos vídeos de teste (bola atrás das pernas, saindo e voltando ao quadro) são tratadas como a "oclusão proposital" exigida.
   - **Estimativa pelas marcações** (aproximada, porque a marcação é a cada 5 quadros e mistura oclusão com bola fora do quadro): **8 trechos sem a bola visível com duração de 10–30 quadros no lateral, 3 no centro.**
   - **Oclusão simulada como complemento:** apagar as detecções da bola por 10–30 quadros em trechos com a bola visível. Mede exatamente a recuperação, porque a verdade de campo é conhecida.
3. **O que substitui cada cena:**

| Cena prevista | Substituto com os vídeos de teste |
|---|---|
| V0 xadrez (intrínsecos) | Calibração de Zhang do iPhone 15 já feita pelo Gabriel (Atividade 01); distância focal do vídeo estimada na cena ([009](009-geometria-do-campo-e-filtro-em-metros.md)) |
| V1 campo com marcos | Medidas do gol feitas pelo Gabriel (escanteio–trave 5 m, trave–trave 3 m) + traves verticais e paredes do campo como restrições ([009](009-geometria-do-campo-e-filtro-em-metros.md)) |
| V2 bola parada (σm) | Trechos em que a bola fica parada nos vídeos de teste (ex.: início do lateral, ~2 s) |
| V3 velocidade de referência | **Cortado** (era o experimento extra E5) |
| V4–V6 passes, oclusão, jogo | Os próprios vídeos de teste |

## Consequências

- **Riscos para a avaliação:**
  - o enunciado fala em oclusão "proposital", e usamos naturais. Se o professor exigir, um clipe curto (30–60 s, alguém parando na frente da bola) resolve, sem calibração;
  - as medidas de campo são poucas, então a calibração do vídeo do centro depende de priors ([009](009-geometria-do-campo-e-filtro-em-metros.md)).
- **Plano de trabalho:** o bloco "A — gravação definitiva" do [006](006-estado-atual-e-proximos-passos.md) deixa de ser o caminho crítico.
- **Uso de IA:** documentado à parte pela dupla. Os commits não levam atribuição de IA (regra no `CLAUDE.md`).
