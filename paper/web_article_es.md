Un modelo de lenguaje de tipo Mixture-of-Experts con 30 000 millones de parámetros (Qwen3-30B-A3B) corre sobre **cuatro placas Raspberry Pi 5** —aproximadamente **€500 de hardware solo CPU**, sin GPU, sin NPU— a **15,143 tok/s de decode**: bit-exact y **un +16,1% por encima del resultado público mejor documentado** para este modelo y esta clase de hardware (13,04 tok/s, b4rtaz #255). Esta es la versión completa y nativa para web de nuestro informe técnico; el [PDF original y todos los datos en bruto](https://github.com/hellomatik-org/distributed-llama/tree/kernel-opt-t02/paper) son públicos.

## Resultados de un vistazo

| Métrica | Valor |
|---|---|
| **Throughput de decode** (comparación directa frente al techo público) | **15,143 tok/s** |
| Mejora sobre b4rtaz #255 (13,04 tok/s, decode) | **+16,1%** |
| Throughput de serving sostenido (prefill incluido) | 14,449 tok/s |
| Time-to-first-token (TTFT) | 557 ms |
| Ancho de banda DRAM por nodo (sostenido / techo del fabricante) | 11,4 / 17 GB/s |
| Hardware / coste | 4× Pi 5 16GB / ~€500 |
| Ganancia con telemetría apagada en decode (A/B emparejado) | +5,18% |
| Restricciones respetadas | bit-exact, sin overclock, sin cambio de modelo |

```chart
{"type":"bar","xKey":"system","yDomain":[0,16.6],"highlightLast":"#1f9d57","unit":"tok/s",
"series":[{"key":"decode","label":"Decode tok/s","color":"#9aa3b2"}],
"data":[{"system":"b4rtaz #255 (record público)","decode":13.04},{"system":"Este trabajo (clean-room)","decode":15.143}],
"caption":"Mismo modelo y misma clase de hardware (Qwen3-30B-A3B Q40, 4× Raspberry Pi 5), métrica de decode: una mejora bit-exact del +16,1% (n=20, telemetría apagada)."}
```

## Por qué esto es difícil

La inferencia en el edge de grandes modelos de lenguaje sobre ordenadores de placa única de consumo se estudia cada vez más como alternativa a la nube: preserva la privacidad y tiene bajo coste. La Raspberry Pi 5 es el SBC ARM más extendido con suficiente RAM (16 GB) para alojar modelos cuantizados, y su Gigabit Ethernet permite formar pequeños clústeres. Pero la Pi 5 **no tiene un acelerador aprovechable** —la GPU VideoCore VII carece de compute shaders de propósito general—, así que la inferencia corre en la CPU, donde la ejecución está **limitada por el ancho de banda de memoria**.

Adoptamos **distributed-llama** v0.16.5 con **doce cambios a nivel de código fuente** desarrollados en este trabajo (ocho parches del framework más cuatro optimizaciones bit-exact de kernel y fusión de operaciones), además de tuning persistente del kernel en tiempo de ejecución. La trayectoria va desde un baseline denso de Llama-3.1-8B a 5,70 tok/s, pasando por un baseline de Qwen3-MoE a 11,40 tok/s, hasta el resultado de decode de 15,143 tok/s.

### Cinco restricciones duras

Cada optimización de este trabajo obedece a cinco reglas, descubiertas de forma progresiva y adoptadas como reglas de diseño. Estas reglas descartan la mayoría de las aceleraciones de la literatura reciente —y eso es precisamente lo que hace que las optimizaciones supervivientes sean desplegables en producción con **cero riesgo de regresión de calidad**:

- **Salida bit-exact** — el SHA-256 de los primeros 100 token-ids generados coincide con una referencia fija (`seed=42`, `temperature=0`).
- **Sin recompilar el kernel** — kernel de Pi OS de fábrica 6.12.75.
- **Sin cambiar el modelo** — Qwen3-30B-A3B Q40 está fijado.
- **Sin overclock de CPU** — el silicio se mantiene en los 2,4 GHz nominales.
- **Sin reducción de calidad** — top-k sin cambios en 8, sin re-cuantizar a la baja, sin podar expertos.

## Diseño del sistema

### Hardware (por nodo)

- **SoC:** Broadcom BCM2712 (4× Arm Cortex-A76 @ 2,4 GHz, ARMv8.2-A)
- **Memoria:** 16 GB LPDDR4X @ 4267 MT/s, ancho de banda teórico ≈ **17 GB/s**
- **Almacenamiento:** SSD NVMe vía PCIe Gen 2 (≈700 MB/s de lectura)
- **Red:** Gigabit Ethernet integrado; latencia intra-clúster medida 0,226 ms
- **SO:** Debian 13 *trixie*, kernel 6.12.75 aarch64
- **Sin acelerador aprovechable** (la GPU V3D carece de compute shaders para LLM; sin NPU)

El clúster es un coordinador **root** (`rpi-1005`, que además sirve la API HTTP) más tres **workers**, conectados por Gigabit Ethernet full-duplex, con sincronización tensor-parallel por cada capa del transformer.

### Stack de software

- **Motor de inferencia:** distributed-llama v0.16.5 + 8 parches del framework + 4 optimizaciones bit-exact de kernel/fusión de operaciones.
- **Modelo:** Qwen3-30B-A3B Q40 (30B parámetros totales, 128 expertos, top-k = 8, ~3B activos por token).
- **Allocator:** jemalloc 2 precargado vía `LD_PRELOAD`.
- **Front-end HTTP:** un proxy Python propio que sanea las peticiones de clientes con OpenAI estricto.

## Metodología

Seguimos las convenciones de MLPerf Inference v5.1. Protocolo: **dos runs de calentamiento (descartados)**, **n = 20 runs de medición** para la configuración principal, prompt fijo, `temperature = 0` para determinismo, un único cliente, carga de fondo en reposo. Las estadísticas se reportan como media, mediana, desviación estándar, intervalo de confianza al 95% y percentiles p50/p90/p99. **Toda afirmación de mejora se valida bit-exact** mediante hashing SHA-256 de la secuencia de token-ids generada frente a una referencia fija, salvo que se indique explícitamente lo contrario.

## La trayectoria de optimización

```chart
{"type":"bar","xKey":"stage","yDomain":[0,16.4],"highlightLast":"#1f9d57","unit":"tok/s","height":460,
"series":[{"key":"toks","label":"tok/s","color":"#2f6fed"}],
"referenceLines":[{"y":13.04,"label":"techo 13.04","color":"#d98a00","position":"insideTopLeft"}],
"data":[{"stage":"Llama 3.1 8B (denso)","toks":5.70},{"stage":"Cambio a Qwen3-30B MoE","toks":11.40},{"stage":"max-seq + swap limpio","toks":12.71},{"stage":"Stage 8 parches+flags","toks":13.72},{"stage":"Stage 9 sysctls TIER-0","toks":14.011},{"stage":"Stage 10 fusión SILU·MUL","toks":14.081},{"stage":"Stage 13 1-pasada real","toks":14.27},{"stage":"Stage 14–15 chunk+NIC","toks":14.449},{"stage":"Stage 17 decode clean-room","toks":15.143}],
"caption":"Trayectoria de optimización y resultado final en clean-room. Las barras 1–8 reportan throughput sostenido; la barra verde final es la réplica clean-room de #255 en la Stage 17 sobre la métrica de decode. Línea discontinua: el techo público."}
```

### La mayor ganancia individual: denso → MoE

El cambio de mayor impacto fue migrar de **Llama 3.1 8B** (denso, ~5 GB de pesos Q40, todos los parámetros activos por token) a **Qwen3-30B-A3B** (MoE, 128 expertos, top-k = 8, ~3 GB de pesos activos por token). El throughput limitado por ancho de banda mejoró un **59%** (7,18 → 11,4 tok/s) sin ningún otro cambio. MoE *es* una forma de activación dispersa: para top-8-de-128 la ratio de activación es 8/128 = 6,25% de los pesos de expertos más los parámetros compartidos, y el multiplicador efectivo de ancho de banda coincide casi exactamente con la aceleración observada.

### Ocho parches del framework

Los parches corrigen bugs críticos y desbloquean flags de optimización. Lo más destacado:

| # | Parche | Por qué importaba |
|---|---|---|
| 1 | `NnByte` → `NnUint` para `nBatches` | Overflow de `uint8_t`: `nbatches=256` se convertía silenciosamente en 0 (256 mod 256), disparando una aserción en la capa de embedding. |
| 2 | Forzar `finish_reason` = `stop`/`length` | Un `finish_reason` vacío metía a los clientes OpenAI estrictos en bucles de reintento infinitos. |
| 3 | `try/catch` alrededor de `json::parse` | Cuerpos malformados lanzaban excepciones no capturadas, abortando el daemon (SIGABRT). |
| 6 | `posix_memalign(64, n)` para los pipes | El `new[]` por defecto en ARM64 está alineado a 16 B; la vectorización NEON requiere alineación a línea de caché (64 B). |
| 7 | TCP `SO_RCVBUF`/`SO_SNDBUF` = 8 MB | Los buffers por defecto de 208 KiB provocaban bloqueo en escritura bajo la sincronización en ráfaga al final de cada capa. |

### Tuning de kernel y SO (Stage 9, TIER 0)

Una ronda estructurada de investigación encontró ajustes de red y memoria a nivel de SO que valían una mejora medible y sin tocar el código. GRO (Generic Receive Offload) agrupa paquetes, añadiendo 50–200 µs de latencia; para las ráfagas de sincronización de 510 kB del paso all-reduce sobre una LAN de 1 GbE eso es puro overhead, así que lo desactivamos. Subimos `rmem_max`/`wmem_max` desde el valor por defecto de 256 kB para que las ventanas TCP puedan crecer por encima del tamaño de la ráfaga, y bajamos `vm.swappiness` a 1 para mantener residentes los pesos hechos `mlock`. Efecto neto: **13,720 → 14,011 tok/s** (+2,12%).

Esta Stage en solitario explica **el 1,5% de la mejora final del 16,1%**, y es un hallazgo crítico: **Debian 13 de fábrica no está configurado para inferencia limitada por ancho de banda de memoria**. Los parámetros por defecto del kernel asumen cargas de trabajo de propósito general (servidores web, bases de datos, shells interactivas) donde la coalescencia de paquetes y la disposición para swap son valores por defecto sensatos. Para inferencia en el edge en hardware de consumidor, estos valores por defecto perjudican activamente el rendimiento. Customizar el kernel Linux sin recompilación —vía configuración de `sysctl` y tuning de NIC en tiempo de ejecución persisted como una unidad `systemd`— es así *no* una optimización marginal: es **infraestructura crítica** para alcanzar un throughput competitivo en esta clase de hardware. Este hallazgo aplica ampliamente a cualquier carga de trabajo distribuida limitada por ancho de banda de memoria en SBCs ARM, y subraya por qué el despliegue containerizado pre-tuned (una única configuración de kernel cuidada distribuida a todos los nodos) es ya requisito indispensable para la inferencia en el edge en producción.

### Dos ganancias bit-exact contraintuitivas

Las dos ganancias a nivel de código más decisivas van **a contracorriente del saber recibido** sobre la misma base de código:

1. **Eliminar el software prefetch (Stage 12, +1,05%).** El bucle interno del matmul NEON+dotprod arrastraba dos llamadas a `__builtin_prefetch`. Barrimos cinco variantes y descubrimos que *eliminar* ambas era la única ganadora: el hardware prefetcher del Cortex-A76 (detección de stride sobre el acceso secuencial a pesos) es más eficiente que las pistas manuales, que compiten por slots de emisión en el decodificador de 4 vías.

2. **Una fusión SILU·MUL de una sola pasada de verdad (Stage 13, +1,5%).** La fusión anterior (Stage 10) era un envoltorio fino de dos llamadas que mantenía el resultado intermedio residente en memoria: 3 cargas + 2 almacenamientos por elemento. La reescribimos como un único bucle NEON que mantiene los valores en registros de principio a fin —2 cargas + 1 almacenamiento por elemento, una reducción de ~33% en operaciones de memoria para este kernel— preservando el bit-exact (la secuencia aritmética es idéntica; solo se elimina la reescritura del intermedio).

```text
Algorithm — Stage 13 single-pass silu_mul_F32 (bit-exact NEON fusion)
Input:  output buffer y[0..n-1] (in-place), multiplier m[0..n-1], thread t of T
Output: y[i] <- silu(y[i]) * m[i]
Invariant: arithmetic (vrecpeq_f32 + one Newton-Raphson + multiply) is
           byte-identical to the prior two-pass code; only the intermediate
           writeback between the two passes is removed.

 1  (start, end) <- split_threads(n, T, t)
 2  for i = start to end-4 step 4 do          // bucle interno NEON, 4 lanes
 3      x  <- vld1q_f32(y + i)                 // 1 carga vectorial
 4      mv <- vld1q_f32(m + i)                 // 1 carga vectorial
 5      e  <- expf_neon(-x)
 6      d  <- 1 + e
 7      r  <- vrecpeq_f32(d)                   // estimacion del reciproco
 8      r  <- r * (2 - d * r)                  // 1 iteracion Newton-Raphson
 9      silu <- x * r
10      out  <- silu * mv
11      vst1q_f32(y + i, out)                  // 1 store; sin escritura intermedia
12  end for
13  for remaining i < end do                   // cola escalar
14      y[i] <- (y[i] / (1 + e^-y[i])) * m[i]
15  end for
```

### Tamaño de chunk de red y tuning de NIC en tiempo de ejecución (Stages 14–15)

El bucle all-reduce `writeMany`/`readMany` topaba cada syscall `send()`/`recv()` en 4 KB. Con los buffers TCP de 8–32 MB ajustados en la Stage 9, eso genera 4× más syscalls de las necesarias; ensanchar a 16 KB redujo el número de syscalls a la cuarta parte (+1,25%). Un bundle de NIC en tiempo de ejecución (anillo RX agrandado, Receive Flow Steering fuera de la CPU 0, NAPI diferida), persistido como una unidad `systemd`, sumó otro +1,99% combinado. Ambos son bit-exact por construcción: solo cambia la planificación de la NIC, no la aritmética del modelo.

## Resultados: throughput y estabilidad

El clúster es excepcionalmente estable run a run —un coeficiente de variación de solo **0,52%**. La distribución de más abajo es la muestra de n = 20 en un snapshot intermedio de la Stage 6 (media **12,708 tok/s**); la configuración final alcanza **14,449 tok/s** (mejor run 14,557, IC 95% ±0,038), con las etapas posteriores subiendo el throughput sin cambiar este comportamiento de latencia.

```chart
{"type":"line","xKey":"run","yDomain":[12.4,13.0],"unit":"tok/s","height":340,
"series":[{"key":"toks","label":"tok/s","color":"#2f6fed"}],
"referenceLines":[{"y":12.708,"label":"media 12.708","color":"#1f9d57"}],
"data":[{"run":1,"toks":12.85},{"run":2,"toks":12.81},{"run":3,"toks":12.74},{"run":4,"toks":12.74},{"run":5,"toks":12.69},{"run":6,"toks":12.67},{"run":7,"toks":12.72},{"run":8,"toks":12.78},{"run":9,"toks":12.62},{"run":10,"toks":12.68},{"run":11,"toks":12.64},{"run":12,"toks":12.77},{"run":13,"toks":12.58},{"run":14,"toks":12.69},{"run":15,"toks":12.64},{"run":16,"toks":12.71},{"run":17,"toks":12.76},{"run":18,"toks":12.65},{"run":19,"toks":12.69},{"run":20,"toks":12.71}],
"caption":"Throughput por run a lo largo de 20 runs de medición (snapshot de la Stage 6, calentamientos excluidos). Coeficiente de variación 0,52%: un clúster muy estable."}
```

El time-to-first-token es de **557 ms de media** (p50 545 ms, máx 638 ms). El throughput escala con la longitud de la respuesta hasta una asíntota: las respuestas más cortas están dominadas por el TTFT:

```chart
{"type":"line","xKey":"max_tokens","logX":true,"yDomain":[11.5,13],"unit":"tok/s","height":320,
"series":[{"key":"toks","label":"tok/s sostenido","color":"#F76B1C"}],
"data":[{"max_tokens":50,"toks":11.83},{"max_tokens":100,"toks":12.47},{"max_tokens":200,"toks":12.67},{"max_tokens":400,"toks":12.86},{"max_tokens":800,"toks":12.62}],
"caption":"Throughput sostenido frente a longitud de respuesta (x logarítmica). Converge cerca de 12,86 tok/s para respuestas ≥ 400 tokens."}
```

**El prefill es el límite práctico.** La tasa de prefill se mantiene por encima de 15 tok/s hasta prompts de 2K tokens, pero para prompts de 20K tokens (p. ej. system prompts grandes de agentes) el tiempo de prefill proyectado supera los 20 minutos: la cota superior de usabilidad para cargas de trabajo completas de agentes en esta configuración.

### Memoria y térmica

El root carga buffers extra para la orquestación y el serving HTTP; los workers conservan >9 GB de margen para más crecimiento de la KV-cache. Bajo carga sostenida todos los nodos se sitúan en 54–56 °C con **cero throttling** (umbral de throttle 85 °C).

```chart
{"type":"bar","xKey":"node","stacked":true,"unit":"GB","height":340,"yDomain":[0,16],
"series":[{"key":"used","label":"Usado (modelo + buffers)","color":"#2f6fed"},{"key":"available","label":"Disponible","color":"#A8C68A"}],
"data":[{"node":"rpi-1005 (root)","used":12,"available":3.0},{"node":"rpi-1006","used":6.5,"available":9.4},{"node":"rpi-1007","used":6.3,"available":9.5},{"node":"rpi-1008","used":6.3,"available":9.5}],
"caption":"Utilización de memoria por nodo durante inferencia sostenida."}
```

## A dónde se va el tiempo: el muro de memoria

El profiling con el PMU de ARM (`perf stat` sobre 60 s de inferencia sostenida) localiza el cuello de botella sin ambigüedad: **49% de ciclos detenidos en el backend** y **11,4 GB/s de tráfico DRAM sostenido por nodo**, ~67% del techo del fabricante de ~17 GB/s. Las CPU esperan a la memoria aproximadamente la mitad del tiempo.

```chart
{"type":"bar","xKey":"phase","layout":"horizontal","stacked":true,"unit":"%","height":210,"xDomain":[0,100],
"series":[{"key":"matmul","label":"Matmul Q40 (MoE)","color":"#2f6fed"},{"key":"sync","label":"Barrera de sync","color":"#d98a00"},{"key":"syscalls","label":"Syscalls (send/recv)","color":"#9aa3b2"},{"key":"other","label":"Orquestación y otros","color":"#c7cfdb"}],
"data":[{"phase":"per-token","matmul":56,"sync":17,"syscalls":4.5,"other":22.5}],
"caption":"Desglose del wall-clock por token. El matmul Q40 domina; la barrera de sincronización es el único margen abordable por software."}
```

| Métrica del PMU | Root (rpi-1005) | Worker (rpi-1006) |
|---|---:|---:|
| stalled-cycles-backend (% de ciclos) | **49,25%** | 47,69% |
| Ancho de banda de lectura DRAM | 2,74 GB/s | 2,61 GB/s |
| Ancho de banda de escritura DRAM | 8,66 GB/s | 8,47 GB/s |
| Total DRAM por nodo | 11,40 GB/s | 11,08 GB/s |
| IPC | 1,74 | — |
| Tasa de miss dTLB / iTLB | 0,08% / 0,01% | — |

Las tasas sub-0,1% de TLB y de fallo de predicción de saltos confirman que la configuración de páginas de 16 KB ya es óptima (las transparent hugepages no ayudarían). `objdump` muestra 322 instrucciones de producto escalar NEON `udot`/`sdot` en el bucle interno a IPC 1,74: el kernel ya está vectorizado hasta el límite del producto escalar de ARMv8.2-A. **Por eso todo experimento del lado de cómputo devuelve cero ganancia: los núcleos están ociosos esperando a la DRAM, no faltos de slots de emisión.**

El techo es físico. En escala logarítmica, la LPDDR4X de la Pi 5 está ~16× por debajo de un Apple M4 Pro y ~200× por debajo de una H100:

```chart
{"type":"bar","xKey":"platform","layout":"horizontal","logX":true,"unit":"GB/s","height":300,
"series":[{"key":"bw","label":"Memory bandwidth (GB/s)","color":"#1b2a4a"}],
"data":[{"platform":"Pi 5 LPDDR4X","bw":17},{"platform":"Mac M4 Pro","bw":273},{"platform":"RTX 3060","bw":360},{"platform":"Mac M3 Ultra","bw":800},{"platform":"H100 SXM5","bw":3350}],
"caption":"Ancho de banda de memoria entre plataformas (escala logarítmica). Este es el techo físico que alcanza el clúster."}
```

## El hallazgo de la telemetría (una confirmación limpia)

Durante el barrido final en clean-room encontramos dos agentes de monitorización de fondo corriendo en los cuatro nodos (la "carga de fondo en reposo" de nuestra metodología anterior). En una carga limitada por ancho de banda de memoria, estos no salen gratis: recorren periódicamente las estadísticas de memoria del sistema, consumiendo ancho de banda de DRAM que la fase de decode necesita. Un A/B emparejado:

| Fondo | n | Decode (tok/s) | Prefill (tok/s) | vs 13,04 |
|---|---:|---:|---:|---:|
| Telemetría ON (Alloy + cAdvisor) | 10 | 14,397 ± 0,153 | 18,74 | +10,4% |
| **Telemetría OFF (clean-room)** | 20 | **15,143 ± 0,097** | 18,81 | **+16,1%** |

Parar los dos agentes subió el decode un **+5,18%** (los IC no se solapan), mientras que **el prefill no cambió** (limitado por cómputo, alta intensidad aritmética). Es también una lección práctica: los agentes de observabilidad co-ubicados gravan silenciosamente la inferencia limitada por memoria, y un nodo de producción monitorizado rinde varios puntos porcentuales por debajo de un benchmark en clean-room.

## Stage 16: una barrera WFE/SEV

La barrera entre pasos hacía busy-spin sobre un atómico con una pista `yield` de ARM, de modo que tres hilos en espera releían continuamente una línea de caché que el hilo que avanza escribe: tráfico de coherencia que compite con el hilo que dirige la E/S de red. Sustituimos el spin por el mecanismo de eventos de ARMv8: los que esperan emiten `wfe` (espera de bajo consumo a un evento) y el hilo que avanza emite un broadcast `sev`. La corrección se apoya en el registro de eventos "pegajoso" de ARM, con el flujo de eventos del temporizador arquitectado como red de seguridad de despertar periódico. Es solo señalización, así que bit-exact por construcción.

```chart
{"type":"bar","xKey":"build","unit":"tok/s","yDomain":[14.0,14.6],"highlightLast":"#1f9d57","errorKey":"ci","height":320,
"series":[{"key":"mean","label":"mean tok/s","color":"#9aa3b2"}],
"data":[{"build":"yield spin (Stage 15)","mean":14.261,"ci":0.034},{"build":"WFE/SEV (Stage 16)","mean":14.329,"ci":0.042}],
"caption":"Barrera WFE/SEV de la Stage 16 frente al yield-spin, A/B emparejado en frío y misma sesión (n=40/brazo). +0,48%, bit-exact, Welch t=2,45, p=0,014. Barras de error: IC 95%."}
```

El cambio se conserva: bit-exact, gratis en tiempo de ejecución, nunca más lento, y reduce el consumo y el calor de los núcleos en espera.

## Lo que no funcionó

En consonancia con las normas de reproducibilidad, documentamos todo intento por tratar (intent-to-treat). Una selección de los 26 callejones sin salida catalogados:

| Configuración | Causa raíz del fallo |
|---|---|
| Llama 3.3 70B Q40 en 4× Pi 5 | 38 GB de pesos fuerzan swap agresivo; 0,15 tok/s con thrashing. |
| Framework EXO | Depende de Apple MLX (Metal + Neural Engine + UMA); no compila en ARM Linux. |
| prima.cpp | El descubrimiento de topología por ZMQ se cuelga >10 min en la Pi 5; nunca arranca. |
| llama.cpp + RPC | Regresión de 25× frente a un solo nodo (paralelismo de pipeline dominado por el overhead de red). |
| Repack ARM I8MM / SMMLA | El Cortex-A76 **no** implementa I8MM (`grep -c i8mm /proc/cpuinfo` = 0). I8MM es una característica de A78+. |
| Transparent hugepages, jumbo frames, intercalado NUMA, PGO | Probados y revertidos: neutros o dañinos en esta carga limitada por DRAM. |
| Software prefetch (PLDL2KEEP+L1 por niveles) | −0,47%; el HW prefetcher ya gana (ver Stage 12). |

La razón recurrente de inaplicabilidad es estructural: las técnicas modernas de mayor palanca requieren un kernel más nuevo, una característica de hardware de la que carece el A76, o un reinicio/recompilación que descartamos, que es precisamente lo que hace desplegables sobre hardware de fábrica las ganancias bit-exact supervivientes.

> **"ARM" no es una categoría de hardware.** Apple Silicon, los SBC Cortex-A76 y los núcleos Neoverse de clase servidor son tres plataformas distintas, con una dispersión de ancho de banda de ~30× y conjuntos de características completamente diferentes (sin I8MM, sin GEMM FP16, sin SVE en el A76). Frameworks supuestamente escritos para "ARM Linux" pueden requerir implícitamente cómputo en GPU, sincronización específica del hardware o extensiones de ISA, y estas distinciones solo afloran en tiempo de ejecución.

## Coste / rendimiento en contexto

| Setup | tok/s (clase 8B) | Coste (aprox.) |
|---|---:|---:|
| **4× Pi 5 16GB + dllama MoE (este trabajo, decode)** | **15,143** | ~€500 |
| 1× Mac Mini M4 8GB + MLX | 25 | 599 USD |
| 1× NVIDIA Jetson Orin Nano Super | 21,75 | 249 USD |
| 1× desktop + RTX 3060 12GB | ~40 | ~€700 |

El clúster de Pi no es el tok/s más barato, pero ofrece inferencia totalmente on-premise sin coste por token y con un margen considerable de reposo para cargas de trabajo co-ubicadas.

## Advertencias honestas (amenazas a la validez)

- **El titular mezcla efectos.** El +16,1% compara nuestro clúster optimizado de 16 GB contra el clúster vanilla de 8 GB de b4rtaz (y `--nthreads 3` frente a 4). Es una *comparación de sistema extremo a extremo*, no una atribución aislada de la aceleración a nuestro código. Las magnitudes aisladas limpiamente que sí podemos defender sin confound son los A/B sobre el mismo hardware: el delta de telemetría (+5,18%) y los deltas de SO/`nthreads`. El benchmark vanilla-vs-nuestro sobre el *mismo* silicio de 16 GB es la siguiente medición.
- **Bit-exact ≠ IEEE-754 estricto.** "Bit-exact" significa igualdad SHA-256 de los token-ids generados bajo los flags de build fijos del proyecto (que incluyen `-ffast-math`); es igualdad frente a nuestro propio baseline canónico, no frente a un build arbitrario de terceros.
- **No se midió la energía.** Los julios por token son la única métrica estándar del edge que no reportamos (no se disponía de un vatímetro de enchufe); el protocolo está esbozado.
- **Un único modelo, un único sitio.** Todas las cifras son para Qwen3-30B-A3B Q40 detrás de un switch Gigabit con nodos de un mismo lote.

## Conclusión: el muro de memoria

En la métrica de decode que usa el techo público, el clúster alcanza **15,143 tok/s** (±0,097, n = 20, telemetría apagada), **un +16,1% por encima de los 13,04 tok/s** documentados públicamente para Qwen3-30B-A3B Q40 en 4× Pi 5; el serving sostenido extremo a extremo (prefill incluido) es de 14,449 tok/s. Hasta donde sabemos, esta es la tasa más alta a la que se ha hecho funcionar un modelo Mixture-of-Experts de 30 000 millones de parámetros en esta clase de hardware: bit-exact, sin overclock y sin pérdida de calidad de salida.

Con un 49% de ciclos detenidos en el backend y ~11,4 de ~17 GB/s realizados por nodo, el decode está limitado por el ancho de banda de memoria: los bytes movidos por token los fija el modelo y su cuantización, y **no queda disponible ninguna reducción bit-exact adicional de ese tráfico**. La única palanca de software que queda —**Async Tensor Parallelism** (~250 LOC sobre la base ya incluida en el repositorio)— solapa la comunicación con el cómputo en lugar de mover menos bytes. El experimento de telemetría afila el argumento: incluso el pequeño porcentaje de ancho de banda de DRAM que consume un monitor de fondo es directamente visible en la tasa de decode, mientras que el prefill, limitado por cómputo, queda intacto.

En resumen, hemos alcanzado el **muro de memoria** que presenta la Raspberry Pi 5. Superarlo ya no es un problema de software sino de silicio: requiere memoria con más ancho de banda.

## Recursos

- **Código, PDF del paper y datos en bruto:** [github.com/hellomatik-org/distributed-llama](https://github.com/hellomatik-org/distributed-llama/tree/kernel-opt-t02) (rama `kernel-opt-t02`, `paper/`).
- **El record público con el que comparamos:** [discusión #255 de b4rtaz/distributed-llama](https://github.com/b4rtaz/distributed-llama/discussions/255).
- **Motor upstream:** [b4rtaz/distributed-llama](https://github.com/b4rtaz/distributed-llama).
