# 001 — Vídeos originais (.MOV 1080p) e processamento em 640 px

**Data:** 21/11/2026

## Contexto

Os primeiros vídeos de teste chegaram como `.mp4` de 848×480 (`campo_lateral_low.mp4` e `campo_centro_low.mp4`). Nessa resolução, a bola no fundo do campo fica com uns 5 px. Depois chegaram os originais do celular.

| Arquivo | Resolução | Codec | fps | Quadros | Duração |
|---|---|---|---|---|---|
| `campo_lateral.MOV` | 1920×1080 | HEVC | 29,994 | 4169 | 139,0 s |
| `campo_centro.MOV` | 1920×1080 | HEVC | 29,994 | 3722 | 124,1 s |
| `campo_*_low.mp4` | 848×480 | H.264 | 29,99 | iguais | iguais |

## Decisão

1. **Usar os `.MOV` originais como fonte, sem converter.**
   - **Leitura:** o OpenCV 5.0 (pip, com FFmpeg) lê HEVC direto, decodificando ~180 fps em 1080p.
   - **Mesmos quadros:** os `.MOV` têm o mesmo número de quadros dos `_low`, então os índices batem.
2. **Processar em 640 px de largura**, mantendo a proporção: 1920×1080 → 640×360.
   - O enunciado pede 640×480 a ≥ 15 fps; como o vídeo é 16:9, 640×360 é o equivalente (até menos pixels).
3. **Marcações em pixels do vídeo original.** Cada ferramenta converte para a sua resolução de processamento.

## Alternativas

- **Continuar com os `_low`:** descartado.
  - **Recorte com zoom:** os `_low` são recortes do original, com campo de visão menor. Isso apareceu na comparação lado a lado do quadro 3708: o `.MOV` mostra bem mais parede e gol.
  - **Resolução:** a bola fica menor no fundo do campo.
- **Converter os .MOV para H.264:** desnecessário, porque a leitura direta funciona e é rápida.
- **Processar em resolução maior** (1280 px): fica como opção para um modo "recorte guiado" (seção 5.4 do escopo). O detector a 1280 roda a ~60 fps contra ~300 fps a 640.

## Evidência

As cores do `.MOV` vêm mais claras e menos saturadas que as do `_low` (mediana HSV da grama na mesma cena):

| | H | S | V |
|---|---|---|---|
| `.MOV` lateral | 35 | 101–103 | 133–135 |
| `_low` lateral | 36 | 126 | 105 |
| `.MOV` centro | 38 | 96–106 | 108–139 |
| `_low` centro | 40 | 137 | 89 |

Isso obrigou a tornar adaptativos os limiares de "branco" do detector (ver [002](002-detector.md)).

## Consequências

- O README lista os `.MOV` como vídeos principais; os `_low` ficam só para comparação.
- Os vídeos ficam fora do git (`videos/` no `.gitignore`), com link para o Google Drive no README.
- A bola mede de ~3 a ~30 px de diâmetro em 640 px de largura, dependendo da distância e do borrão de movimento.
