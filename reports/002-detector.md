# 002 — Detector da bola: branco sobre a grama, sem subtração de fundo

**Data:** 21/11/2026 · **Código:** `src/detector.py` · **Testes:** `tests/test_detector.py`

## Contexto

- **A bola** é branca com detalhes coloridos (vermelho, azul) e quase sempre rasteira. A ideia original de uma bola laranja ou amarela não se confirmou.
- **A cena:** campo de grama à noite, com refletores, parede branca ao fundo, jogadores descalços (pele clara sob a luz), folhas e papéis no gramado e a rede do gol.
- **A câmera** foi segurada na mão (ver [003](003-camera-na-mao-estabilizacao.md)).

## Decisão

Detector só por aparência, a cada quadro:

1. **Grama:** máscara HSV com H ∈ [25, 60], S ≥ 50 e V ≥ 30.
2. **Borda do campo:** em cada coluna, a primeira linha (de cima para baixo) em que a grama domina (densidade ≥ 0,6 numa janela vertical de 9 px). Para tirar os jogadores que tampam a borda, mediana de 5 colunas e depois uma **abertura morfológica 1D** (mínimo seguido de máximo) com janela de 121 colunas. A abertura remove "picos" mais estreitos que a janela e preserva bordas inclinadas. Tudo abaixo da borda é campo.
3. **Branco adaptativo**, relativo à grama do próprio quadro:
   - S < min(70; 0,5 · S_mediana da grama);
   - V > max(120; V_mediana da grama + 30).
4. **Candidatos = branco ∩ campo**, seguido de um fechamento elíptico de 5 px (preenche os detalhes coloridos da bola) e de uma abertura de 3 px (tira o ruído da grama).
5. **Componentes conectados** filtrados por:
   - área de 8 a 900 px²;
   - circularidade ≥ 0,40;
   - razão de aspecto ≤ 2,0;
   - fração de branco dentro da mancha ≥ 0,5;
   - **fração de grama no anel em volta ≥ 0,5**. Esse teste descarta o rodapé, a rede e manchas grudadas em jogadores.
6. **Score** = circularidade × brancura × grama no anel. Ele é usado na inicialização e na associação da trilha.

## Alternativas (testadas, nessa ordem)

| Versão | Ideia | Resultado | Por que mudou |
|---|---|---|---|
| v0a | Branco fixo (S < 80, V > 135) ∩ campo, sem anel | ~6 candidatos/quadro, muitos na linha entre o rodapé branco e a grama | Faltava distinguir "branco cercado de grama" |
| v0b | Manchas "não grama" no campo + brancura + anel | 1,7–3,3 cand./quadro, mas a bola se **funde com a sombra e com o pé**, sobretudo quando borrada | Perdia a bola nos lances mais importantes |
| v0c | Branco + fechamento + anel de grama (atual) | Acha a bola borrada e perto da câmera (lateral 3708, centro 740) | — |
| v0c + borda por mediana | Borda do campo por mediana de 61 colunas | A borda "contornava" jogadores grandes perto da câmera | Trocada pela abertura 1D (janela 121) |
| v0c + branco fixo nos `.MOV` | Limiar S < 80 / V > 135 | **24–28 candidatos/quadro**: a grama clara do `.MOV` (S ≈ 70–80, V ≈ 175) passava como "branco" | Limiares adaptativos (a bola tem S ≈ 18, V ≈ 216) |

**Subtração de fundo (MOG2):** implementada (`use_mog2`) mas **desligada**. Com a câmera na mão o modelo de fundo não se sustenta; ela só volta a fazer sentido sobre a imagem estabilizada (ver próximos passos).

## Evidência

- **Velocidade:** ~300 fps a 640 px (~3 ms por quadro).
- **Contra a verdade de campo**, lateral, 810 quadros marcados (510 com a bola visível), raio de 10 px:

| Métrica | Valor |
|---|---|
| Bola entre os candidatos (recall) | **69,2%** |
| Melhor candidato (maior score) é a bola | 38,2% |
| Candidatos falsos por quadro | 5,99 |
| Erro do centroide (mediana) | 2,39 px |

Comando: `python -m tools.analyze_video videos/campo_lateral.MOV --match-radius 10`.

## Consequências e limitações

- **Falsos positivos que sobram:**
  - pés e canelas claros;
  - folhas e papéis brancos no gramado;
  - a base da rede do gol;
  - a bola encostada na parede branca, que se perde porque a parede também é branca.
- **Perdas da bola:**
  - colada no pé (o anel de grama falha);
  - quando sai do quadro, de propósito, nos vídeos de teste.
- **Na gravação definitiva:** limpar o gramado (folhas e papéis) antes de gravar.
- **O detector entrega vários candidatos:** quem escolhe a bola é a trilha ([005](005-gating-e-trilha.md)). Com ~6 falsos por quadro, a qualidade da associação depende muito de reduzir esse número ([006](006-estado-atual-e-proximos-passos.md)).

## Revisão (21/11) — comparação com detectores de blob

Testados como geradores alternativos de candidatos, com o mesmo teste do anel de grama (ver [007](007-candidatos-falsos-e-detectores.md)):

| Gerador | Bola entre os candidatos | Falsos por quadro | ms/quadro |
|---|---|---|---|
| LoG em escala | 64,9% | 36,7 | 26 |
| SimpleBlobDetector | 25,1% | 0,31 | 5 |
| **HSV (este)** | **69,2%** | 5,99 | 3,5 |

**O HSV continua.** Os falsos positivos passam a ser tratados por uma etapa de score combinado (movimento compensado, LSS e prior de tamanho), não por troca de detector.
