<!-- paper-url: https://github.com/hellomatik-org/distributed-llama/blob/kernel-opt-t02/paper/main_en.pdf -->

Un modelo Mixture-of-Experts de 30 000 millones de parámetros (Qwen3-30B-A3B) corre sobre **cuatro placas Raspberry Pi 5** (hardware solo CPU, sin GPU, sin NPU) a **15,143 tok/s de decode**: bit-exact y **un +16,1% por encima del mejor resultado público documentado** para este modelo y esta clase de hardware (13,04 tok/s, b4rtaz #255). Esto es una versión técnica condensada de nuestro informe; el paper completo (cada tabla, figura y log en bruto) está a un clic.

## Resultados de un vistazo

| Métrica | Valor |
|---|---|
| Throughput de decode (frente al techo público) | **15,143 tok/s** |
| Mejora sobre b4rtaz #255 (13,04 tok/s, decode) | **+16,1%** |
| Throughput de serving sostenido (prefill incluido) | 14,449 tok/s |
| Time-to-first-token (TTFT) | 557 ms |
| Ancho de banda DRAM por nodo (sostenido / techo del fabricante) | 11,4 / 17 GB/s |
| Hardware | 4× Pi 5 16GB |
| Restricciones respetadas | bit-exact, sin overclock, sin cambio de modelo |

```chart
{"type":"bar","xKey":"system","yDomain":[0,16.6],"highlightLast":"#1f9d57","unit":"tok/s",
"series":[{"key":"decode","label":"Decode tok/s","color":"#9aa3b2"}],
"data":[{"system":"b4rtaz #255 (record público)","decode":13.04},{"system":"Este trabajo (clean-room)","decode":15.143}],
"caption":"Mismo modelo y misma clase de hardware (Qwen3-30B-A3B Q40, 4× Raspberry Pi 5), métrica de decode: una mejora bit-exact del +16,1% (n=20)."}
```

## Por qué es difícil

La inferencia en el edge de LLMs sobre ordenadores de placa única se estudia cada vez más como alternativa a la nube: privada y barata. La Raspberry Pi 5 es el SBC ARM más extendido con RAM suficiente (16 GB) para alojar modelos cuantizados, y su Gigabit Ethernet permite formar pequeños clústeres. Pero la Pi 5 **no tiene acelerador aprovechable** (la GPU VideoCore VII carece de compute shaders de propósito general), así que la inferencia corre en la CPU, donde el decode está **limitado por el ancho de banda de memoria**.

Adoptamos **distributed-llama** v0.16.5 con **doce cambios a nivel de código fuente** desarrollados en este trabajo (ocho parches del framework más cuatro optimizaciones bit-exact de kernel y fusión de operaciones), además de tuning persistente del kernel en runtime.

### Cinco restricciones duras

Cada optimización obedece a cinco reglas adoptadas como reglas de diseño. Descartan la mayoría de las aceleraciones lossy de la literatura, y eso es justo lo que hace desplegable lo que sobrevive, con **cero riesgo de regresión de calidad**:

- **Salida bit-exact**: el SHA-256 de los primeros 100 token-ids generados coincide con una referencia fija (`seed=42`, `temperature=0`).
- **Sin recompilar el kernel**: kernel de Pi OS de fábrica 6.12.75.
- **Sin cambiar el modelo**: Qwen3-30B-A3B Q40 está fijado.
- **Sin overclock de CPU**: el silicio se mantiene en los 2,4 GHz nominales.
- **Sin reducción de calidad**: top-k fijo en 8, sin re-cuantizar a la baja, sin podar expertos.

## Diseño del sistema

**Por nodo:** Broadcom BCM2712 (4× Cortex-A76 @ 2,4 GHz, ARMv8.2-A); 16 GB LPDDR4X @ 4267 MT/s (≈17 GB/s teóricos); NVMe vía PCIe Gen 2; Gigabit Ethernet integrado (latencia intra-clúster medida 0,226 ms); Debian 13, kernel 6.12.75 aarch64; sin acelerador aprovechable. El clúster es un coordinador **root** (que además sirve la API HTTP) más tres **workers**, Gigabit Ethernet full-duplex, con sincronización tensor-parallel por cada capa del transformer.

**Software:** distributed-llama v0.16.5 + nuestros parches; modelo Qwen3-30B-A3B Q40 (30B total, 128 expertos, top-k = 8, ~3B activos/token); jemalloc 2 vía `LD_PRELOAD`; un pequeño proxy Python que sanea las peticiones de clientes con OpenAI estricto.

## Metodología

Seguimos las convenciones de MLPerf Inference v5.1: **dos runs de calentamiento (descartados)**, **n = 20 runs de medición**, prompt fijo, `temperature = 0`, un único cliente. Estadísticas como media, mediana, desviación estándar, IC 95% y percentiles p50/p90/p99. **Toda afirmación de mejora se valida bit-exact** mediante SHA-256 de la secuencia de token-ids frente a una referencia fija, salvo que se indique lo contrario.

## La trayectoria de optimización

```chart
{"type":"bar","xKey":"stage","yDomain":[0,16.4],"highlightLast":"#1f9d57","unit":"tok/s","height":440,
"series":[{"key":"toks","label":"tok/s","color":"#2f6fed"}],
"referenceLines":[{"y":13.04,"label":"techo 13.04","color":"#d98a00","position":"insideTopLeft"}],
"data":[{"stage":"Llama 3.1 8B (denso)","toks":5.70},{"stage":"Cambio a Qwen3-30B MoE","toks":11.40},{"stage":"max-seq + swap limpio","toks":12.71},{"stage":"Stage 8 parches+flags","toks":13.72},{"stage":"Stage 9 sysctls TIER-0","toks":14.011},{"stage":"Stage 10 fusión SILU·MUL","toks":14.081},{"stage":"Stage 13 1-pasada real","toks":14.27},{"stage":"Stage 14–15 chunk+NIC","toks":14.449},{"stage":"Stage 17 decode clean-room","toks":15.143}],
"caption":"Trayectoria de optimización y resultado final en clean-room. Las barras 1–8 reportan throughput sostenido; la barra verde final es la réplica clean-room de #255 en la Stage 17 sobre la métrica de decode. Línea discontinua: el techo público."}
```

### El cambio con mayor efecto: denso → MoE

El cambio con mayor efecto individual fue migrar de **Llama 3.1 8B** (denso, ~5 GB de pesos Q40, todos los parámetros activos por token) a **Qwen3-30B-A3B** (MoE, 128 expertos, top-k = 8, ~3 GB activos por token). El throughput limitado por ancho de banda mejoró un **59%** (7,18 → 11,4 tok/s) sin ningún otro cambio. MoE *es* activación dispersa: para top-8-de-128 la ratio de activación es 8/128 = 6,25% de los pesos de expertos más los parámetros compartidos, y el multiplicador efectivo coincide casi exactamente con la aceleración observada.

### Ocho parches del framework

Los parches corrigen bugs críticos y desbloquean flags de optimización. Lo más destacado:

| # | Parche | Por qué importaba |
|---|---|---|
| 1 | `NnByte` → `NnUint` para `nBatches` | Overflow de `uint8_t`: `nbatches=256` se convertía en 0 (256 mod 256), disparando una aserción en la capa de embedding. |
| 2 | Forzar `finish_reason` = `stop`/`length` | Un `finish_reason` vacío metía a los clientes OpenAI estrictos en bucles de reintento infinitos. |
| 3 | `try/catch` alrededor de `json::parse` | Cuerpos malformados lanzaban excepciones no capturadas, abortando el daemon (SIGABRT). |
| 6 | `posix_memalign(64, n)` para los pipes | El `new[]` por defecto en ARM64 alinea a 16 B; NEON requiere alineación a línea de caché (64 B). |
| 7 | TCP `SO_RCVBUF`/`SO_SNDBUF` = 8 MB | Los buffers por defecto de 208 KiB provocaban bloqueo en escritura bajo la sincronización en ráfaga. |

### Dos cambios bit-exact contraintuitivos

Ambas van a contracorriente del saber recibido sobre la misma base de código:

1. **Eliminar el software prefetch (Stage 12, +1,05%).** El bucle interno del matmul NEON+dotprod tenía dos `__builtin_prefetch`. De cinco variantes barridas, *eliminar* ambas fue la única que ayudó: el hardware prefetcher del Cortex-A76 (detección de stride sobre acceso secuencial) supera a las pistas manuales, que compiten por slots de emisión en el decodificador de 4 vías.

2. **Una fusión SILU·MUL de una sola pasada de verdad (Stage 13, +1,5%).** La fusión anterior era un envoltorio de dos llamadas que mantenía el intermedio en memoria (3 cargas + 2 stores por elemento). La reescribimos como un único bucle NEON que mantiene los valores en registros (2 cargas + 1 store por elemento) preservando el bit-exact (aritmética idéntica; solo se elimina la reescritura del intermedio).

![Antes (two-pass) vs después (single-pass): la fusión SILU·MUL elimina el store/load intermedio entre los dos bucles. 2 loads + 1 store por iteración en vez de 3 + 2, bit-exact.](/api/contents/research/0001_19d46624-4ce1-4a02-b01d-db23007f1cc5/images/silu-mul-fusion.png)

### Tamaño de chunk de red y tuning de NIC en runtime (Stages 14–15)

El bucle all-reduce `writeMany`/`readMany` topaba cada syscall `send()`/`recv()` en 4 KB. Con los buffers TCP mayores de la Stage 9 eso son 4× más syscalls de las necesarias; ensanchar a 16 KB las redujo a la cuarta parte (+1,25%). Un bundle de NIC en runtime (anillo RX agrandado, Receive Flow Steering fuera de la CPU 0, NAPI diferida), persistido como unidad `systemd`, sumó +1,99% combinado. Ambos son bit-exact: solo cambia la planificación de la NIC, no la aritmética del modelo.

## Customizar el Linux importó tanto como el código

Es tentador atribuir un resultado así al motor de inferencia. En el mismo silicio de 16 GB, **la mayor parte de la aceleración desplegable vino de re-tunear el sistema operativo anfitrión tan agresivamente como la aplicación**, y cada palanca de abajo respeta la restricción bit-exact, así que ninguna cuesta calidad:

- **`SDRAM_BANKLOW=1`** en la EEPROM del bootloader: un cambio de intercalado de memoria a nivel de firmware que subió el ancho de banda efectivo de lectura de 8,3 a 12,5 GB/s por nodo. **+5,9%**, bit-exact, sin overclock.
- **`--nthreads 3` en lugar de 4**, liberando un núcleo, con la IRQ de Ethernet y el Receive Packet Steering fijados a ese núcleo libre (CPU 3), más **jemalloc** (arena/tcache afinados). Juntos **+10,3%**.
- **Governor de CPU `performance`** a 2,4 GHz, y **pesos en `mlock`** (~4,45 GB bloqueados/worker) para que el modelo nunca pagine.
- **Stack de red / VM (Stage 9, TIER 0)**: TCP BBR, `SO_RCVBUF`/`SO_SNDBUF` agrandados, **GRO desactivado** (añade 50–200 µs de puro overhead a las ráfagas all-reduce de 510 kB sobre 1 GbE), `busy_poll`, `vm.swappiness=1`. **+2,12%**.

Todo el estado queda congelado en unidades `systemd`, de modo que un clúster en frío arranca directo a la configuración optimizada, sin pasos manuales, totalmente reproducible. La conclusión es general: en una carga limitada por ancho de banda de memoria **el sistema operativo no es un sustrato neutro.** El intercalado de memoria por firmware, la colocación de núcleos/IRQ, el allocator, la residencia de páginas y el stack de red mueven cada uno la aguja en porcentajes de un dígito que se acumulan, y en el mismo hardware suman *más que los propios cambios de código fuente*.

## Resultados: throughput y estabilidad

La varianza run a run es baja: un coeficiente de variación del **0,52%**. La distribución de más abajo es la muestra de n = 20 en un snapshot intermedio de la Stage 6 (media **12,708 tok/s**); la configuración final alcanza **14,449 tok/s** (mejor run 14,557, IC 95% ±0,038), con las etapas posteriores subiendo el throughput sin cambiar este comportamiento de latencia.

```chart
{"type":"line","xKey":"run","yDomain":[12.4,13.0],"unit":"tok/s","height":340,
"series":[{"key":"toks","label":"tok/s","color":"#2f6fed"}],
"referenceLines":[{"y":12.708,"label":"media 12.708","color":"#1f9d57"}],
"data":[{"run":1,"toks":12.85},{"run":2,"toks":12.81},{"run":3,"toks":12.74},{"run":4,"toks":12.74},{"run":5,"toks":12.69},{"run":6,"toks":12.67},{"run":7,"toks":12.72},{"run":8,"toks":12.78},{"run":9,"toks":12.62},{"run":10,"toks":12.68},{"run":11,"toks":12.64},{"run":12,"toks":12.77},{"run":13,"toks":12.58},{"run":14,"toks":12.69},{"run":15,"toks":12.64},{"run":16,"toks":12.71},{"run":17,"toks":12.76},{"run":18,"toks":12.65},{"run":19,"toks":12.69},{"run":20,"toks":12.71}],
"caption":"Throughput por run a lo largo de 20 runs de medición (snapshot de la Stage 6, calentamientos excluidos). Coeficiente de variación 0,52%."}
```

El time-to-first-token es de **557 ms de media** (p50 545 ms, máx 638 ms). El throughput escala con la longitud de la respuesta hasta una asíntota: las respuestas cortas están dominadas por el TTFT:

```chart
{"type":"line","xKey":"max_tokens","logX":true,"yDomain":[11.5,13],"unit":"tok/s","height":320,
"series":[{"key":"toks","label":"tok/s sostenido","color":"#F76B1C"}],
"data":[{"max_tokens":50,"toks":11.83},{"max_tokens":100,"toks":12.47},{"max_tokens":200,"toks":12.67},{"max_tokens":400,"toks":12.86},{"max_tokens":800,"toks":12.62}],
"caption":"Throughput sostenido frente a longitud de respuesta (x logarítmica). Converge cerca de 12,86 tok/s para respuestas ≥ 400 tokens."}
```

**El prefill es el límite práctico.** La tasa de prefill se mantiene por encima de 15 tok/s hasta prompts de 2K tokens, pero para prompts de 20K (system prompts grandes de agentes) el prefill proyectado supera los 20 minutos: la cota superior de usabilidad para cargas completas de agentes aquí.

Bajo carga sostenida todos los nodos se sitúan en 54–56 °C con **cero throttling** (umbral 85 °C). El root carga buffers extra de orquestación/serving; los workers conservan >9 GB de margen para la KV-cache.

```chart
{"type":"bar","xKey":"node","stacked":true,"unit":"GB","height":320,"yDomain":[0,16],
"series":[{"key":"used","label":"Usado (modelo + buffers)","color":"#2f6fed"},{"key":"available","label":"Disponible","color":"#A8C68A"}],
"data":[{"node":"rpi-1005 (root)","used":12,"available":3.0},{"node":"rpi-1006","used":6.5,"available":9.4},{"node":"rpi-1007","used":6.3,"available":9.5},{"node":"rpi-1008","used":6.3,"available":9.5}],
"caption":"Utilización de memoria por nodo durante inferencia sostenida."}
```

## A dónde se va el tiempo: el muro de memoria

El profiling con el PMU de ARM (`perf stat`, 60 s de inferencia sostenida) clava el cuello de botella: **49% de ciclos detenidos en el backend**, **11,4 de ~17 GB/s** de DRAM por nodo, IPC 1,74. Las tasas sub-0,1% de TLB y fallo de predicción confirman que las páginas de 16 KB ya son óptimas; `objdump` muestra 322 `udot`/`sdot` NEON en el bucle interno, ya en el límite de ARMv8.2-A. Los núcleos esperan a la DRAM, por eso todo experimento del lado de cómputo devuelve cero ganancia.

```chart
{"type":"bar","xKey":"phase","layout":"horizontal","stacked":true,"unit":"%","height":210,"xDomain":[0,100],
"series":[{"key":"matmul","label":"Matmul Q40 (MoE)","color":"#2f6fed"},{"key":"sync","label":"Barrera de sync","color":"#d98a00"},{"key":"syscalls","label":"Syscalls (send/recv)","color":"#9aa3b2"},{"key":"other","label":"Orquestación y otros","color":"#c7cfdb"}],
"data":[{"phase":"per-token","matmul":56,"sync":17,"syscalls":4.5,"other":22.5}],
"caption":"Desglose del wall-clock por token. El matmul Q40 domina; la barrera de sincronización es el único margen abordable por software."}
```

El techo es físico. En escala logarítmica, la LPDDR4X de la Pi 5 está ~16× por debajo de un Apple M4 Pro y ~200× por debajo de una H100:

```chart
{"type":"bar","xKey":"platform","layout":"horizontal","logX":true,"unit":"GB/s","height":300,
"series":[{"key":"bw","label":"Memory bandwidth (GB/s)","color":"#1b2a4a"}],
"data":[{"platform":"Pi 5 LPDDR4X","bw":17},{"platform":"Mac M4 Pro","bw":273},{"platform":"RTX 3060","bw":360},{"platform":"Mac M3 Ultra","bw":800},{"platform":"H100 SXM5","bw":3350}],
"caption":"Ancho de banda de memoria entre plataformas (escala logarítmica). Este es el techo físico que alcanza el clúster."}
```

## El hallazgo de la telemetría

Durante el barrido final en clean-room encontramos dos agentes de monitorización de fondo corriendo en los cuatro nodos (la "carga de fondo en reposo" de nuestra metodología anterior). En una carga limitada por ancho de banda, no salen gratis: recorren periódicamente las estadísticas de memoria del sistema, consumiendo ancho de banda de DRAM que el decode necesita.

| Fondo | n | Decode (tok/s) | Prefill (tok/s) |
|---|---:|---:|---:|
| Monitorización ON | 10 | 14,397 ± 0,153 | 18,74 |
| **Monitorización OFF (clean-room)** | 20 | **15,143 ± 0,097** | 18,81 |

Parar los agentes subió el **decode un +5,18%** (los IC no se solapan) dejando el **prefill sin cambios**, una doble disociación: el decode está limitado por ancho de banda y el prefill, limitado por cómputo, es el control negativo. Es también una lección práctica: la observabilidad co-ubicada grava silenciosamente la inferencia memory-bound, y un nodo de producción monitorizado rinde varios puntos por debajo de un benchmark en clean-room.

## Stage 16: una barrera WFE/SEV

La barrera entre pasos hacía busy-spin sobre un atómico con una pista `yield`, de modo que tres hilos releían continuamente una línea de caché que el hilo que avanza escribe: tráfico de coherencia que compite con el hilo que dirige la E/S de red. Sustituimos el spin por el mecanismo de eventos de ARMv8: los que esperan emiten `wfe` (espera de bajo consumo) y el que avanza emite un broadcast `sev`. Es solo señalización, así que bit-exact por construcción; además baja consumo y calor en los núcleos en espera.

```chart
{"type":"bar","xKey":"build","unit":"tok/s","yDomain":[14.0,14.6],"highlightLast":"#1f9d57","errorKey":"ci","height":320,
"series":[{"key":"mean","label":"mean tok/s","color":"#9aa3b2"}],
"data":[{"build":"yield spin (Stage 15)","mean":14.261,"ci":0.034},{"build":"WFE/SEV (Stage 16)","mean":14.329,"ci":0.042}],
"caption":"Barrera WFE/SEV de la Stage 16 frente al yield-spin, A/B emparejado en frío y misma sesión (n=40/brazo). +0,48%, bit-exact, Welch t=2,45, p=0,014. Barras de error: IC 95%."}
```

## Lo que no funcionó

En consonancia con las normas de reproducibilidad, documentamos todo intento. Una selección de los 26 callejones sin salida catalogados:

| Configuración | Causa raíz del fallo |
|---|---|
| Llama 3.3 70B Q40 en 4× Pi 5 | 38 GB de pesos fuerzan swap agresivo; 0,15 tok/s con thrashing. |
| Framework EXO | Depende de Apple MLX (Metal + Neural Engine + UMA); no compila en ARM Linux. |
| prima.cpp | El descubrimiento de topología por ZMQ se cuelga >10 min en la Pi 5; nunca arranca. |
| llama.cpp + RPC | Regresión de 25× frente a un solo nodo (pipeline dominado por overhead de red). |
| Repack ARM I8MM / SMMLA | El Cortex-A76 **no** implementa I8MM (`grep -c i8mm /proc/cpuinfo` = 0); es de A78+. |
| Transparent hugepages, jumbo frames, NUMA interleave, PGO | Probados y revertidos: neutros o dañinos en esta carga limitada por DRAM. |
| Software prefetch (PLDL2KEEP+L1 por niveles) | −0,47%; el HW prefetcher ya gana (ver Stage 12). |

La razón recurrente de inaplicabilidad es estructural: las técnicas modernas de mayor palanca necesitan un kernel más nuevo, una característica de hardware de la que carece el A76, o un reinicio/recompilación que descartamos, que es justo lo que hace desplegables sobre hardware de fábrica las ganancias bit-exact supervivientes.

## Advertencias y limitaciones

- **El titular +16,1% mezcla efectos**: nuestro clúster optimizado de 16 GB frente al vanilla de 8 GB de b4rtaz (y `--nthreads 3` vs 4). En **silicio idéntico de 16 GB**, vanilla → nuestra build → totalmente tuneado es 13,15 → 13,88 → **15,143** (+15,2%), de los cuales solo **+5,6% es atribuible al código fuente**; el resto es la elección de modelo, el flag de firmware, la configuración runtime y un entorno limpio. La Tabla 22 del paper lo descompone palanca a palanca.
- **"Bit-exact" significa SHA-256 frente a nuestra propia build canónica** (los flags incluyen `-ffast-math`), no IEEE-754 estricto. La energía (J/token) es la única métrica estándar del edge que aún no medimos. Todas las cifras son para un modelo, un sitio y un lote de silicio.

## Conclusión y trabajo futuro

El número del titular importa menos que el método detrás. **La mayor parte de la aceleración no vino del modelo ni de kernels novedosos, sino de tratar el sistema operativo como una parte ajustable del stack de inferencia**: intercalado de memoria por firmware, colocación de núcleos e IRQ, el allocator, la residencia de páginas y un stack de red afinado para el tráfico all-reduce entre nodos, en ráfagas y de baja latencia. Nada de esto tocó el hardware, el reloj ni la calidad de salida del modelo. Esa es la lección transferible: **este tipo de optimización a nivel de OS y de red aplica a cualquier sistema de inferencia distribuida en CPU, no solo a distributed-llama.**

Hemos alcanzado el muro de memoria que presenta la Pi 5: los bytes por token los fija el modelo y su cuantización, y no queda reducción bit-exact adicional de ese tráfico. La palanca de software restante, Async Tensor Parallelism, solapa comunicación con cómputo en lugar de mover menos bytes.

**Qué viene.** Planeamos publicar una **imagen de Linux ajustada para inferencia distribuida** (un OS pre-configurado para baja latencia entre nodos en clústeres tipo dllama) para que estas ganancias estén disponibles *out of the box*, **sin cambiar el hardware, hacer overclock ni bajar la calidad del modelo.** También planeamos pasar la misma optimización de OS y red por **otros modelos open-source**, para comprobar hasta dónde generaliza el enfoque más allá de Qwen3-MoE.

## Recursos

- **Paper, código y datos en bruto:** [github.com/hellomatik-org/distributed-llama](https://github.com/hellomatik-org/distributed-llama/tree/kernel-opt-t02) (rama `kernel-opt-t02`, `paper/`).
- **La referencia pública con la que comparamos:** [discusión #255 de b4rtaz/distributed-llama](https://github.com/b4rtaz/distributed-llama/discussions/255).
