# Documentación técnica — DeepPack3D adaptado a la paletización de cajas de flores en un Boeing 747-400F

> Generado a partir de una revisión completa de todos los archivos de la carpeta (código fuente, notebooks, datos de entrada, PDFs de referencia y metadatos de empaquetado).
> Objetivo: explicar qué hace cada archivo, qué llama a qué, en qué celdas ocurre lo importante, y sobre todo **cómo el paquete genérico DeepPack3D fue adaptado al problema real** de cargar cajas de flores en las 65 posiciones físicas de un avión de carga 747-400F.

---

## 1. El problema que se está resolviendo

**DeepPack3D** (Tsang, Mo, Chung & Lee, 2025 — ver `A deep reinforcement learning approach for online and concurrent 3D bin packing...pdf` y `DeepPack3D A Python package...pdf`) es un paquete genérico de *3D bin packing* pensado originalmente para paletización robótica: llegan cajas de tamaño variable por una "cinta transportadora" (conveyor) y hay que decidir, caja por caja, en qué contenedor (bin) y en qué coordenada (x,y,z) colocarla, ya sea con una red neuronal entrenada por refuerzo (RL) o con heurísticas clásicas de *rectangle/cuboid packing*.

El problema real de este proyecto es distinto y más restringido que el caso genérico:

- Las "cajas" son **cajas de flores cortadas para exportación**, con un catálogo **fijo** de ~18 tamaños estándar en decímetros (ver `Registros nuevas cajas.txt`), no cajas de tamaño arbitrario.
- Los "bins" no son contenedores cúbicos idénticos: son las **65 posiciones de carga reales de un Boeing 747-400F** (pallets M-1, pallets M-1H, pallets P6P de bodega inferior, contenedores LD-1 y el compartimento bulk), cada una con su propio ancho/alto/profundidad, tomadas de la ficha técnica `747-400 F.pdf` (Atlas Air).
- El objetivo no es aprender una política general, sino **generar un plan de estiba concreto por vuelo** (14 vuelos con matrícula `5Y-XXXX`, prefijo de Kenya Airways/Atlas Air), maximizando la utilización volumétrica de cada posición y minimizando cajas sin embarcar.

Todo el trabajo de adaptación consistió en: (a) modificar el motor de DeepPack3D para soportar **bins de tamaños heterogéneos** en una secuencia fija (en vez de bins idénticos), (b) alimentar el sistema con datos reales de manifiestos de carga en vez de datos sintéticos, y (c) usar únicamente las heurísticas (no la RL, que quedó como experimento inconcluso) para producir los planes de estiba finales por vuelo.

---

## 2. Mapa de dependencias (quién importa/llama a quién)

```
747-400 F.pdf  ─────────────┐ (fuente de los tamaños reales de bins)
Registros nuevas cajas.txt ─┼──► input_*.txt / Input_worst_*.txt / input_int_5Y-*.txt
                             │      (catálogos y manifiestos de cajas)
                             ▼
geometry.py  (Cuboid: geometría de cajas/espacios)
      ▲
      │
SpacePartitioner.py (empaquetador de UN bin: splits libres + height_map)
      ▲                                   binpacker.py (código gemelo,
      │                                    NO se usa, ver §9.2)
split_gen.py (generador sintético de cajas vía cortes guillotina/no-guillotina)
      ▲
      │
conveyor.py (Conveyor / FileConveyor / InputConveyor: fuente de items)
      ▲
      │
env.py → MultiBinPackerEnv (entorno: N bins abiertos, estado, acciones, reward)
      ▲
      │
agent.py → Agent (RL, red Q) y HeuristicAgent (bl/baf/bssf/blsf)
      ▲
      │
deeppack3d.py → función generadora deeppack3d(...) + CLI (main())
      ▲
      │
Notebooks: Heuristicas.ipynb, "Heuristica - BL/BAF/BLSF.ipynb",
           Testing_MIT.ipynb, Testing_MIT_2.ipynb
```

Regla general: cada módulo hace `from <módulo_anterior> import *`, así que el "más alto" en la cadena (los notebooks, luego `deeppack3d.py`, luego `agent.py`) ve todos los símbolos de los niveles inferiores.

---

## 3. El motor genérico de bin-packing (paquete DeepPack3D original)

### 3.1 `geometry.py` — primitiva geométrica

Define `Cuboid(x, y, z, width, height, depth)`: un paralelepípedo con propiedades derivadas (`left/right/bottom/top/back/front`, `volume`, `size`, `coord`).

Métodos clave:
- `intersect(other, edge=False)` — solapamiento estricto (`edge=False`) o inclusivo de bordes (`edge=True`).
- `contain(other)` — si `self` contiene completamente a `other`.
- `split(other, maximal=True)` — **corazón del algoritmo de particionado de espacio libre**: dado un espacio libre `self` y una caja recién colocada `other` que lo intersecta, genera hasta 6 sub-cuboides (uno por cada cara que sobresale) que representan el espacio libre restante. Con `maximal=True` genera splits que se solapan entre sí (se filtran después); con `maximal=False` genera splits disjuntos (usado solo para calcular `space_utilization`).
- `fit(item)` — comprobación simple de si un tamaño `(w,h,d)` cabe dentro del cuboide.

No fue modificado para la adaptación; es la geometría pura que reutilizan todos los módulos superiores.

### 3.2 `SpacePartitioner.py` (empaquetador realmente usado) y `binpacker.py` (código muerto)

`SpacePartitioner.py:66-165` implementa `SpacePartitioner(size)`, que representa **un solo bin**:
- `free_splits`: lista de cuboides libres (no solapados, mantenidos "maximales") — arranca como `[Cuboid(0,0,0,w,h,d)]`.
- `splits`: lista de cajas ya colocadas.
- `height_map`: matriz 2D `(depth, width)` con la altura acumulada en cada celda (vista de "planta" — heightmap típico de packing).
- `fit(cuboid)` (líneas 78-94): optimización — si hay menos cajas colocadas que splits libres, comprueba colisión contra las cajas colocadas; si no, comprueba que algún split libre *contenga* completamente al cuboide (más barato cuando ya hay muchas piezas).
- `add(cuboid)` (líneas 96-142): coloca la caja, actualiza `height_map` con `np.maximum`, y **reparticiona** el espacio libre: por cada `free_split` que intersecta la nueva caja, lo reemplaza por `partition.split(cuboid)`; luego elimina particiones nuevas que estén contenidas dentro de otra (evita explosión combinatoria de splits redundantes).
- `space_utilization()` (líneas 144-157): suma el volumen ocupado y, para el volumen libre, resta solapamientos entre `free_splits` usando `split(..., maximal=False)` de forma iterativa — es una comprobación de consistencia (`used + free == np.prod(size)`, si no `raise Exception('wtf')`).
- `render()` — dibujo 3D con matplotlib (usado por `agent.py` cuando `visualize=True`).

`binpacker.py` contiene una **clase casi idéntica** `BinPacker` con un algoritmo de `add()` más simple (usa `itertools.combinations` en vez de la lógica de contención por lotes) y una función `first_fit()`. **Ninguna de las dos se usa en ningún otro archivo del proyecto** (se confirmó con `grep`) — es un remanente del paquete original que quedó sin eliminar; todo el pipeline real usa `SpacePartitioner`.

### 3.3 `split_gen.py` — generador sintético de cajas (solo modo `generated`)

Usado únicamente cuando `--data=generated` (no es el camino usado para las flores, pero sí lo usa el entrenamiento RL de prueba). Dos algoritmos recursivos que cortan un contenedor en sub-cajas aleatorias:
- `gullotine_cut` / `_gullotine_cut` (líneas 49-89): corte tipo guillotina clásico (un plano corta todo el volumen de lado a lado).
- `nongullotine_cut` / `_nongullotine_cut` (líneas 91-148): corte "no guillotina", parte siempre el split libre más pequeño en vez de cortar todo el espacio, generando una distribución de tamaños más realista/variada. Este es el generador que usa `Conveyor` internamente (`conveyor.py:103`).
- `reset_rng(seed)` (línea 9): fija la semilla global (`rng`, `np.random.default_rng`) que comparten `split_gen`, `conveyor` y todo lo que dependa de aleatoriedad.

### 3.4 `conveyor.py` — fuente de ítems (la "cinta transportadora")

Todas las clases heredan de `ItemGenerator(k)` (línea 4), que implementa el patrón *lookahead de k ítems*: `peek()` rellena un buffer hasta tener `k` ítems futuros visibles; `grab(n)` extrae y consume el ítem en la posición `n` del lookahead.

- **`FileConveyor(k, path)`** (líneas 27-50): **la clase que realmente alimenta el problema de las flores**. `_iter()` (líneas 38-42) abre el archivo y por cada línea hace `w, h, d = map(int, line.split(' '))` — exactamente el formato de todos los `input_*.txt`. Se activa con `--data=file --path=...` en `deeppack3d.py`.
- `InputConveyor(k)` (líneas 52-82): lee cajas por teclado (`input()`), solo para uso interactivo manual.
- `Conveyor(k, ..., assigned_items=None)` (líneas 84-150): el conveyor "genérico" del paquete original. Si `assigned_items` no es `None` (línea 106-109), simplemente itera esa lista tal cual — **este es el mecanismo que los notebooks de `Testing_MIT.ipynb` usan para inyectar arrays de numpy con cajas reales sin pasar por un archivo de texto** (p. ej. `Conveyor(k=env.k, assigned_items=items_all).reset()`). Si no hay `assigned_items`, genera cajas sintéticas con `nongullotine_cut` vía `split_generator` (línea 103).
- `rotated_sizes(item, rotate=True, remove_duplicate=True)` (líneas 152-170): dado un ítem `(w,h,d)`, devuelve todas las rotaciones válidas en ejes x/y/z (hasta 6, sin duplicados) — usado por `env.py` para generar todas las orientaciones posibles de cada caja antes de buscar dónde colocarla.

### 3.5 `env.py` — el entorno (estado / acciones / recompensa)

Clase base abstracta `Env` (líneas 7-22, no usada directamente) y la clase real **`MultiBinPackerEnv`** (líneas 33-343), que modela *varios bins abiertos simultáneamente* mientras van llegando cajas.

Conceptos clave:
- **Estado** `state()` (líneas 123-138) = `(items, h_maps, actions)`:
  - `items`: los `k` próximos ítems visibles del conveyor (`self.conveyor.peek()`).
  - `h_maps`: lista de `height_map` (uno por bin abierto) — `self._height_maps()` (línea 140-141).
  - `actions`: **todas** las colocaciones factibles ahora mismo, calculadas por `self.actions(...)`.
- **Espacio de acciones** `actions(items, h_maps, rotate, skip)` (líneas 170-191): para cada ítem del lookahead, para cada bin abierto, para cada rotación (`rotated_sizes`), para cada coordenada factible (`placeable_coords`), genera una tupla `(i_bin, (x,y,z), (w,h,d), split)`. Si `skip=False` (`use_skip=False`), solo se consideran acciones para el **primer** ítem del lookahead (fuerza orden FIFO estricto); con `skip=True` (por defecto) se pueden "saltar" ítems del lookahead si no caben, y colocar uno posterior.
- **Coordenadas factibles** `placeable_coords(packer, h_map, size)` (líneas 143-168): recorre los `free_splits` del packer, descarta splits cuyo `top` no llega a la altura total del bin (`packer.size[1]`) o que no puedan contener el tamaño rotado (`split.fit(size)`), y para cada `(x,z)` único calcula `y = max(height_map en esa huella)`. **Restricción de estabilidad** (línea 165): solo se acepta la colocación si más del 50% de la huella de apoyo está exactamente al nivel `y` (`np.count_nonzero(placement == y) / (d*w) > 0.5`) — evita apoyar una caja mayormente "en el aire".
- **Recompensa** `step(action)` (líneas 193-305): coloca la caja, y calcula
  `reward = (pyramid + compactness) / 2`, donde
  `pyramid = volumen_ocupado / suma(height_map)` (premia apilar "en pirámide", plano y bajo) y
  `compactness = volumen_ocupado / (W · max(height_map) · D)` (premia ocupar el prisma envolvente real, sin huecos de altura).
- **Cierre y apertura de bins** (líneas 226-303): cuando un bin se queda sin acciones posibles (`done`), según `self.replace`:
  - `'max'`: reemplaza solo el bin con mayor utilización actual por uno nuevo.
  - `'all'` (el usado en el proyecto): reemplaza **todos** los bins vacíos/agotados de una vez, en bucle, hasta que ya no haya más disponibles o se llegue a `max_bins`.
  - Si ya no se pueden abrir más bins (`self.used_bins + 1 > self.max_bins`), se vuelcan a `used_packers` los bins con contenido y termina el episodio.

**Adaptación específica al problema (lo más importante de este archivo):**

- Parámetro nuevo `bin_sizes=None` en el constructor (línea 48) y en `reset()` (líneas 103-114): si se pasa una lista de tuplas `(W,H,D)`, cada vez que se abre un bin nuevo se usa el **siguiente tamaño de la lista** en vez de un tamaño único global (`self.size`). Esto es lo que permite modelar 65 posiciones físicas distintas del avión como una **cola ordenada de tamaños**, en vez de 65 copias del mismo contenedor.
- `placeable_coords` (línea 143-168) fue ajustado para usar `packer.size[1]` (la altura *real de ese bin concreto*) en vez de `self.size[1]` (que sería el tamaño global único del diseño original) — necesario porque ahora cada packer puede tener una altura distinta.
- `p_map(i_bin, cuboid)` (líneas 307-329) fue reescrito para construir la máscara con la forma exacta de `height_map` del bin `i_bin` (`self.packers[i_bin].size`), en vez de asumir un tamaño global — mismo motivo. (Esta reescritura coincide con los parches `p_map_fixed` que aparecen repetidamente en `Testing_MIT.ipynb`, ver §5.1; el parche ad-hoc del notebook terminó incorporándose de forma permanente aquí.)
- `state(step=False)` deja `h_maps` como **lista de arrays de numpy** (uno por bin, de formas potencialmente distintas) en vez de apilarlas en un único `np.array` 3D — imprescindible porque con bins heterogéneos las `height_map` ya no tienen todas la misma forma y no se pueden apilar. El comentario en el código (línea 126) lo dice explícitamente.
- Lógica de selección del siguiente tamaño al reponer un bin (líneas 245-258 y 274-286): `self.bin_sizes[self.used_bins]` si quedan tamaños en la lista, si no repite el último (`self.bin_sizes[-1]`) como resguardo.

### 3.6 `agent.py` — red Q (RL) y agentes heurísticos

- `q_net(k=1)` (líneas 12-50): red convolucional (Conv2D 64→128→256→512→1024→2048, todas con `BatchNormalization` y regularización L2) que combina tres entradas espaciales (`height_map`, "action map" `p_map`, un mapa constante de unos) más un vector `imap` con los `k-1` ítems restantes del lookahead (aplanado + `Dense(256)`), y produce un único valor Q escalar. Arquitectura DQN.
- **`Agent`** (líneas 52-275): agente de *Deep Q-Learning*.
  - `select(state)` (90-104): ε-greedy — con probabilidad `eps` acción aleatoria, si no, `argmax` de `Q(state)`.
  - `Q_inputs` / `Q` (106-159): construye los tensores de entrada por lotes (`batch_size`) y evalúa la red.
  - `train` / `fit` (168-217): calcula el target de Bellman (`reward + gamma * max Q(next_state)` si no `done`), hace `GradientTape`, aplica gradientes con Adam, y cada `update_epochs` pasos hace un *soft update* de la red objetivo (`q_net`) con un promedio 50/50 respecto a `q_net_target`.
  - `run(max_ep, ...)` (219-275): bucle episodio→pasos; en cada paso `yield`ea la colocación elegida (para que `deeppack3d.py` la reenvíe hacia afuera), y al terminar el episodio hace `yield None` (marcador de "nuevo bin/episodio" que consumen los notebooks) y guarda estadísticas en `self.ep_history`.
  - **Nota de inconsistencia detectada** (`agent.py:107`): `Q_inputs` normaliza usando `W, H, D = self.env.size` — el tamaño *global* de referencia pasado al entorno — en vez del tamaño real del bin `j` (`self.env.packers[j].size`, que sí se usa correctamente en `env.p_map` e `i_map`). Con bins homogéneos esto es inocuo, pero con la lista heterogénea de 65 bins del 747 la normalización de la red Q queda mal calibrada para los bins que no coinciden con `global_size`. Es probablemente una de las razones (junto con el tiempo de entrenamiento) por las que **los notebooks de producción final por vuelo no usan RL, solo heurísticas** (ver §5.4).
- **`HeuristicAgent`** (líneas 277-347): mismo patrón de `run()`, pero en vez de una red Q usa una función heurística (`self.heuristic(actions)`) que recibe **todo** el espacio de acciones factibles y devuelve directamente el índice `(i,j,k)` elegido — sin entrenamiento, sin memoria de repetición, determinista dado el mismo estado.
- **Las 4 heurísticas** (líneas 349-399), todas evalúan **todas** las acciones factibles y ordenan (`sorted`) para quedarse con la de menor "score":
  - `bottom_left` (BL, 349-361): score = `(y+h, x+w, z+d, i, j, k)` — prioriza minimizar la altura final, luego x, luego z. Es el "bottom-left-fill" clásico extendido a 3D.
  - `best_short_side_fit` (BSSF, 363-374): score = `min(W-w, H-h)` del split elegido — minimiza el lado sobrante más corto (deja el hueco "más ajustado" posible).
  - `best_area_fit` (BAF, 376-387): score = `(volumen_del_split, min(W-w,H-h), ...)` — prioriza el split libre de menor volumen que aún quepa la caja, y en empate usa BSSF como desempate.
  - `best_long_side_fit` (BLSF, 389-399): score = `max(W-w, H-h)` — minimiza el lado sobrante más largo.
  
  Estas son la extensión a 3D de las heurísticas clásicas de *rectangle bin packing* (Jylänki, "A Thousand Ways to Pack the Bin"), aplicadas aquí sobre los `free_splits` cuboides.

### 3.7 `deeppack3d.py` — orquestador / CLI / punto de entrada (el archivo más adaptado)

- `parse_args()` (líneas 1-46): define la interfaz CLI (`method`, `lookahead`, `--data`, `--path`, `--n_iterations`, `--seed`, `--verbose`, `--train`, `--batch_size`, `--visualize`).
- `heuristics = {...}` (líneas 58-63): mapa de nombre de método string → función heurística de `agent.py`.
- **`deeppack3d(method, lookahead, ...)`** (líneas 65-176): función **generadora** — es la API tanto de la CLI como de los notebooks. Flujo:
  1. `reset_rng(seed)`.
  2. **Construye `bin_sizes`, la lista fija de 65 tamaños de bin que representan el 747-400F real** (líneas 68-76, ver detalle completo en §4.1-4.2).
  3. Crea `MultiBinPackerEnv(n_bins=len(bin_sizes), size=global_size, max_bins=len(bin_sizes), k=lookahead, bin_sizes=bin_sizes, ...)` (líneas 86-94) — es decir, **abre los 65 bins de una vez** (`n_bins=len(bin_sizes)`) y no permite crear más (`max_bins=len(bin_sizes)`).
  4. Según `data`, sustituye `env.conveyor` por `FileConveyor` (manifiestos reales, `--data=file --path=...`) o `InputConveyor` (`--data=input`); si se deja `generated` usa el `Conveyor` sintético por defecto.
  5. Si `--visualize`, limpia y recrea la carpeta `./outputs/`.
  6. Rama `train=True`: solo válida con `method='rl'`; entrena con `Agent`, con un *epsilon* que decae `0.95` por iteración hasta un mínimo de `0.025`; al final grafica utilización (`util.jpg`) y recompensa (`ep_reward.jpg`) por episodio, y guarda el modelo en `./{uuid}.h5`.
  7. Rama `train=False` (la usada en producción con heurísticas): si `method='rl'` carga `./models/k={lookahead}.h5`; si no, crea un `HeuristicAgent(heuristics[method], env, ...)`. Ejecuta `agent.run(n_iterations, ...)` y va `yield`eando cada colocación tal cual la produce el agente.
  8. **Parche añadido específicamente para este proyecto** (línea 164-165, comentario `# 👉 NUEVO: al final devolvemos el env para poder leer used_packers`): al terminar de iterar, hace un **`yield env`** extra con el objeto `MultiBinPackerEnv` completo. Esto es lo que permite a los notebooks (`Heuristica - BL/BAF/BLSF.ipynb`) reconstruir, después de la corrida, exactamente qué cajas quedaron en cada uno de los 65 bins físicos leyendo `env.used_packers[i].splits` — sin este `yield` no habría forma de recuperar la asignación bin-por-bin una vez terminado el generador.
  9. Si `verbose>0`, imprime tiempo total, próximos ítems pendientes, utilización promedio y bins usados.
- `main()` (líneas 179-197): agota el generador `deeppack3d(...)` desde la CLI (`python deeppack3d.py bl 5 --data=file --path=./input.txt ...`).

---

## 4. La adaptación al problema real (sección central)

### 4.1 Origen de los 65 bins: `747-400 F.pdf`

La ficha `747-400 F.pdf` (hoja de especificaciones de Atlas Air) documenta la configuración de carga de un 747-400F:

| ULD / posición | Dimensiones externas máx. | Cantidad en el avión |
|---|---|---|
| M-1H (cubierta principal, contorneado) | 96×125×118 in (244×317,5×300 cm) | 23 |
| M-1 (cubierta principal) | 96×125×96 in (244×317,5×244 cm) | 5 normales + 2 "especiales" |
| P6P (bodega inferior) | 96×125×64 a 118 in, variable | 9 |
| LD-1 (contenedor, bodega inferior) | 60,4×92,0×64,0 in (153×234×165 cm) | 2 |
| Bulk (bodega inferior, sin pallet) | — | 1 |

`23 + 5 + 2 + 9 + 2 + 1 = 42` posiciones "con nombre", pero como cada M-1H tiene además una franja superior contorneada que sobresale sobre el M-1 estándar, el modelo final descompone esa geometría en **65 posiciones/"bins" discretos** para el motor de packing (ver §4.2). Esta ficha es también la fuente directa de las tablas `REFERENCES` (nombre, cantidad, tamaño en cm) que aparecen en `Testing_MIT.ipynb` (celda 24, 35, 48) usadas en los experimentos intermedios con unidades en centímetros.

### 4.2 Cómo se construye `bin_sizes` (el corazón de la adaptación)

En `deeppack3d.py:71-76` (y replicado, con pequeñas variaciones de detalle, en las celdas de `Heuristicas.ipynb` y en las tres `Heuristica - *.ipynb`):

```python
bin_sizes = []
bin_sizes += [(24, 24, 32)] * 30   # 30 pallets M1     (23 M-1H + 5 M-1 + 2 M-1 especiales)
bin_sizes += [(6, 12, 32)]  * 23   # 23 slices M1H     (franja superior extra de las 23 M-1H)
bin_sizes += [(24, 16, 32)] * 9    # 9 pallets P6P     (bodega inferior)
bin_sizes += [(15, 16, 23)] * 2    # 2 contenedores LD1
bin_sizes += [(20, 20, 18)] * 1    # 1 bulk
# total = 30 + 23 + 9 + 2 + 1 = 65
```

Las dimensiones están en una **unidad entera de rejilla** (no en pulgadas ni cm directamente — son valores reducidos/escalados, probablemente decímetros redondeados o una rejilla proporcional), consistente con que las cajas de flores también estén expresadas en decímetros enteros (ver §4.4). Los 30 "pallets M1" agrupan **todas** las posiciones de cubierta principal (23 M-1H + 5 M-1 + 2 M-1 especiales = 30, cuadra exactamente con la tabla `REFERENCES` de `Testing_MIT.ipynb` celda 24/48), tratando su huella base como homogénea; los "23 slices M1H" modelan aparte la altura adicional que solo tienen los 23 pallets M-1H por encima del M-1 estándar (contorno de fuselaje), como si fuera un segundo "mini-bin" apilable lógicamente sobre el primero. Este `bin_sizes` se pasa a `MultiBinPackerEnv(..., bin_sizes=bin_sizes)`, que es la extensión descrita en §3.5.

Esta misma lista de 65 tamaños se repite **literalmente copiada y pegada** en cada celda de producción de los notebooks de heurísticas (una copia por cada vuelo procesado) — no está centralizada en una función/constante compartida, lo cual es una fuente de riesgo de inconsistencia si algún día cambia (de hecho hay una pequeña discrepancia real: `deeppack3d.py` usa `(20, 20, 18)` para el bulk, mientras que `Heuristicas.ipynb` celda 1 usa `(20, 20, 17)` y las demás celdas y las tres `Heuristica - *.ipynb` vuelven a `(20, 20, 18)` — diferencia de 1 unidad de profundidad, ver §9.1).

### 4.3 El patrón de ejecución "un bin (o todos) por llamada" para planes multi-referencia

Además del camino "65 bins de una sola vez" de `deeppack3d.py`, en `Testing_MIT.ipynb` se desarrolló un patrón alternativo (celdas 27-29, y su variante "grid32" en celdas 44-49) para poder mezclar **distintas referencias de pallet con distinta cantidad cada una**, envolviendo el motor bin por bin:

```python
def pack_one_bin(items_remaining, bin_size, lookahead, mode, ...):
    env = MultiBinPackerEnv(n_bins=1, max_bins=1, size=bin_size, k=lookahead, ...)
    env.conveyor = Conveyor(k=env.k, assigned_items=items_remaining).reset()
    # corre 1 heurística o 1 episodio de Agent(train=True) hasta agotar ese único bin
    # resta del inventario (multiset) las cajas que sí se colocaron
    return placements, items_restantes
```
y luego `run_multi_reference_pipeline` / el bucle de `REFERENCES` (celdas 28-29, 35, 48) itera sobre una lista de referencias `{name, qty, bin_size}`, llamando a `pack_one_bin` `qty` veces por referencia y encadenando el inventario restante de una llamada a la siguiente, exportando un CSV por bin (`export_bin_csv`, columnas `w,h,d,x,y,z`). **Esta ruta terminó siendo un experimento paralelo/preliminar**; la solución final de producción usa en cambio el enfoque de `deeppack3d.py` con `bin_sizes` de 65 posiciones en una sola corrida (§4.2), más simple de mantener y coherente con la lógica ya existente en `env.py` de reponer bins automáticamente.

### 4.4 Los datos de entrada: catálogo de cajas de flores

`Registros nuevas cajas.txt` documenta el **catálogo fijo de tamaños de caja de flores** (formato Largo-Ancho-Alto, en decímetros) y cuántas cajas de cada tamaño se añadieron para construir distintos escenarios de prueba:
- **`Best_1000`**: 17 tamaños distintos, 1000 cajas nuevas repartidas entre ellos (escenario "optimista": mezcla de tamaños favorable al llenado).
- **`Worst_4000`**: mismos tamaños, otra distribución de cantidades (escenario "pesimista": mezcla desfavorable, dominada por cajas planas `7×5×1`).
- **`Worst_500`**: versión reducida del escenario pesimista.

Se verificó contra los datos reales que, por ejemplo, `input_int_5Y-3462_plus2000.txt` solo contiene combinaciones de este catálogo fijo (`6 3 2`, `7 5 1`, `7 5 2`, `8 4 1`, `9 1 1`, `9 2 2`, `9 3 2`, `9 3 3`, `9 5 2`, `10 1 1`, `10 2 1`, `10 2 2`, `10 3 1`, `10 3 2`, `10 3 3`, `10 5 2`, `10 6 2`, `11 3 1`, `11 3 3`) — confirmando que **todos los manifiestos de vuelo usan exclusivamente estos ~19 tamaños de caja reales**, nunca tamaños arbitrarios.

Archivos de datos presentes, por rol:

| Archivo | Líneas (cajas) | Rol |
|---|---:|---|
| `input.txt` | 1 000 | Ejemplo pequeño de la librería original (tamaños "grandes", no del catálogo de flores) — usado en README y pruebas de humo (`Testing_MIT.ipynb` celda 0-1). |
| `input_int.txt` | 10 911 | Escenario base con cajas del catálogo real, en enteros. |
| `input_best.txt` / `input_best_plus1000.txt` | 12 474 / 13 474 | Escenario "mejor caso" (`Best_1000`) y su ampliación con 1000 cajas más. |
| `Input_worst_div10_ceil.txt` | 10 910 | Escenario "peor caso", dimensiones ya divididas entre 10 y redondeadas hacia arriba (de ahí `div10_ceil`) — es decir, conversión de cm reales a decímetros con margen de seguridad. |
| `input_int_plus4000.txt` | 14 910 | Escenario base + 4000 cajas extra (`Worst_4000`), para estresar el llenado con muchas más cajas que espacio disponible. |
| `input_int_5Y-XXXX_plus2000.txt` (×14) | 11 252 – 14 393 | **Manifiestos por vuelo real** (14 vuelos: 3462, 3570, 3570A, 3572, 3580, 3580A, 3582, 3586, 4570, 6572, 6576, 7688, 7734, 8780), cada uno con 2000 cajas adicionales de margen sobre el manifiesto base — son el insumo real de los notebooks de producción (§5.4). |
| `Otros/input_int_plus500.txt`, `Otros/input_int_rand.txt` | 11 411 / 10 911 | Variantes exploratorias adicionales, no referenciadas por el pipeline final. |

Formato de todos estos archivos: una caja por línea, `w h d` separados por espacio, enteros — exactamente lo que `FileConveyor._iter()` (`conveyor.py:38-42`) espera.

### 4.5 Evolución cronológica de la adaptación (según fecha de modificación de archivos)

Los metadatos de archivo permiten reconstruir el orden en que se hizo la adaptación:

1. **2025-10-26** — Clon inicial del paquete DeepPack3D "de fábrica" (`geometry.py`, `SpacePartitioner.py`, `binpacker.py`, `conveyor.py`, `split_gen.py`, `setup.py`): todos con la misma fecha, sin modificar desde entonces.
2. **2025-11-09** — Primer intento de soportar el 747-400F: aparece un archivo (ya no presente como `.py`, solo su bytecode compilado `__pycache__/env_multisize_747_400.cpython-310.pyc`) que era, aparentemente, una **copia separada de `env.py`** dedicada a bins de tamaño múltiple, y en paralelo un módulo externo `pack_with_pallet_plan.py` (tampoco presente como fuente, solo su `.pyc`; vivía en la carpeta hermana `SIMPAC-2024-311-main` referenciada por `Testing_MIT.ipynb`) con la función `pack_stage(...)` usada para orquestar el llenado multi-referencia.
3. **2025-12-04** — La lógica de bins heterogéneos se **fusiona dentro de `env.py`** directamente (parámetro `bin_sizes`), reemplazando el archivo separado del paso 2.
4. **2025-12-07 / 2025-12-08** — Últimos ajustes en `agent.py` (heurísticas) y `deeppack3d.py` (lista fija de 65 bins reales + el `yield env` final) — versión que se usa en los notebooks de producción.

Esto confirma que el camino de "un archivo `env_multisize_747_400.py` aparte" y el módulo externo `pack_with_pallet_plan.py` fueron **prototipos que se abandonaron/fusionaron**; la versión vigente y activamente usada es la de `env.py` + `deeppack3d.py` de esta carpeta.

---

## 5. Los notebooks: qué hace cada uno

### 5.1 `Testing_MIT.ipynb` — bitácora de pruebas y prototipado (50 celdas)

Es un notebook de **trabajo/depuración**, no de producción: mezcla pruebas de humo, parches en caliente al código ya cargado en memoria, y varios experimentos de entrenamiento RL que no llegaron a resultado final. Puntos relevantes:

- **Celdas 0-1**: instalación de dependencias, verificación de Python 3.10, y dos *smoke tests* vía subproceso (`subprocess.run`) llamando a `deeppack3d.py` desde la CLI con el método `bl` y luego `rl`, usando `input.txt`.
- **Celdas 3-4**: demos aisladas de `split_gen.gullotine_cut` y de `SpacePartitioner` con cajas manuales, para entender la API antes de integrarla al problema real.
- **Celdas 5-9**: primeras corridas de `deeppack3d('rl', 5, data='file', path='./input.txt')`, recolectando resultados y graficando utilización por bin (bins homogéneos 32×32×32, todavía sin los bins reales del avión).
- **Celda 10**: **hotfix en caliente** sobre un módulo externo `pack_with_pallet_plan.py` (de la carpeta hermana del capstone) — redefine `pack_stage(...)` para corregir que el generador de `HeuristicAgent.run()` devuelve `(i_bin, (x,y,z), (w,h,d), split)` y no `(item, (x,y,z), ...)` como asumía el código original de ese módulo. Es evidencia directa del proceso de depuración de la integración.
- **Celda 11**: parche de `SpacePartitioner.space_utilization` a una versión "segura" (`_safe_space_utilization`) que evita división por cero / valores fuera de `[0,1]` — se aplica **monkeypatching** directo sobre la clase (`SpacePartitioner.space_utilization = _safe_space_utilization`).
- **Celda 12**: parche de `MultiBinPackerEnv.state` para dejar `h_maps` como lista (no `np.array` apilado) — el mismo cambio que terminó incorporado de forma permanente en `env.py:123-138` (§3.5).
- **Celdas 16-23**: primeros intentos de entrenar RL **con el tamaño de bin real en centímetros** (`BIN_SIZE = (600,250,250)`, luego `(244,318,300)` — el pallet M-1H de la ficha del 747), cargando 10 000 cajas reales desde `input_10k_cm_integer.txt` / `input_boxes_sample_10000_cm_integer.txt` (archivos de esa fase, ya no presentes en esta carpeta). Incluye activación de GPU/`mixed_float16` (celda 23).
- **Celda 21-22**: parche `p_map_fixed` para forzar que la máscara de acción tenga la forma `(D, W)` igual que `height_map` — el mismo fix que terminó en `env.py:307-329` (§3.5), aquí se ve el proceso de descubrir y validar el bug con un `assert`.
- **Celdas 24-29**: primer diseño de la tabla `REFERENCES` (nombre/cantidad/tamaño en cm) y el pipeline `pack_one_bin` + `run_multi_reference_pipeline` descrito en §4.3, ejecutado en modo RL.
- **Celdas 31-49**: segunda ronda de experimentos RL, esta vez **reescalando** las cajas reales en cm a una rejilla fija `GRID=32` (funciones `to_grid` / `to_cm_placement`, celdas 32/37/41) para poder usar la arquitectura de red convolucional de tamaño fijo `(32,32,32)`, entrenando (`k=5` o `k=7`) y guardando modelos como `models/k=5_grid32.h5`, luego corriendo inferencia (`pack_one_bin_inference`) y exportando a CSV. Es la exploración más avanzada del camino RL, pero **no es la que se usó al final** (los notebooks de producción, §5.4, usan heurísticas puras sobre los bins reales sin reescalar a 32³).

En síntesis: `Testing_MIT.ipynb` es el laboratorio donde se **descubrieron y corrigieron** los bugs de `env.py`/`agent.py` al pasar de "un bin cúbico único" a "bins reales heterogéneos", y donde se intentó (sin llegar a producción) un pipeline RL completo.

### 5.2 `Testing_MIT_2.ipynb` — puente hacia el enfoque final (3 celdas)

Notebook corto de transición: abandona el reescalado a cm/grid32 y **vuelve a usar `deeppack3d()` directamente con los bins nativos del proyecto**, ya con `METHOD='bl'` (heurística, no RL) y `PATH='./Input_worst_div10_ceil.txt'`. La celda 2 sí intenta una fase de entrenamino RL breve (`DO_TRAIN=True`, 10 iteraciones) seguida de test, con `K=40` — el lookahead ya se ve subiendo de 5 hacia valores mucho mayores (más tarde estabilizado en torno a 7 en los notebooks finales), reflejo de que con las cajas de flores (muchas piezas relativamente pequeñas) un lookahead corto deja demasiadas colocaciones sin explorar.

### 5.3 `Heuristicas.ipynb` — comparación de las 3 heurísticas sobre los 65 bins reales (21 celdas)

Primer notebook que usa **la lista completa de 65 `bin_sizes_real`** (idéntica a `deeppack3d.py`, copiada inline) y **reconstruye resultados por `bin_id` real** (0-64) en vez de por orden de aparición, iterando `deeppack3d(METHOD, K, data='file', path='./Input_worst_div10_ceil.txt', n_iterations=1)` y clasificando cada colocación con `bins[bin_id].append(...)`. Contiene, en orden, una sección casi idéntica para cada heurística:

- **Celdas 1-11**: sección `Bottom_left` (`METHOD='bl'`, `K=25`) — utilización real por bin, gráfico de barras (64 posiciones), gráfico 3D de un bin cargado, conteo total de cajas empacadas por bin y en todo el avión (**8571 de 10911**, es decir 2340 sin embarcar con esta heurística/lookahead), y desglose de qué tamaños de caja cayeron en cada bin (`Counter`).
- **Celdas 12-14**: sección etiquetada `Best_short_side_fit` pero cuyo código realmente fija `METHOD='baf'` (con `K=90`, luego repetido con `K=50`) — **inconsistencia de rotulado** heredada de copiar/pegar celdas entre secciones (ver §9.1).
- **Celdas 15-17+**: sección `Best_area_fit`, cuyo código a su vez fija `METHOD='bssf'` — mismo tipo de inconsistencia (los nombres de las secciones markdown y el valor real de `METHOD` están cruzados entre BAF y BSSF).
- **Sección final (no mostrada arriba pero presente)**: `Best_long_side_fit` (`blsf`).

Este notebook es exploratorio/comparativo: sirvió para decidir qué heurística y qué lookahead usar antes de pasar a los notebooks de producción por vuelo.

### 5.4 `Heuristica - BL.ipynb`, `Heuristica - BAF.ipynb`, `Heuristica - BLSF.ipynb` — notebooks de producción (55 celdas cada uno)

Son los **notebooks finales que generan el resultado del proyecto**: para cada uno de los tres métodos heurísticos (Bottom-Left, Best-Area-Fit, Best-Long-Side-Fit — nótese que **no existe** un `Heuristica - BSSF.ipynb`, el cuarto método definido en `agent.py` no se llevó a producción por vuelo), corren el mismo patrón sobre **cada uno de los 14 vuelos reales**, con `K=7` fijo en todas las celdas de vuelo (tras haber probado `K=3,4,5` en celdas anteriores de ajuste).

Estructura común (idéntica en los tres archivos, solo cambia el valor de `METHOD` y, por tanto, los números de resultado):

1. **Celdas 1-12**: igual que `Heuristicas.ipynb` pero sobre `input_int.txt` / `input_int_plus4000.txt`, ajustando el lookahead `K` (3, 4, 5) — calibración inicial.
2. **Celda 14 en adelante — cambio clave de método de reconstrucción**: en vez de clasificar por `bin_id` capturado durante la iteración del generador (como hacía `Heuristicas.ipynb`), aquí se **usa el `yield env` final** de `deeppack3d.py` (§3.7 punto 8):
   ```python
   for result in deeppack3d(METHOD, K, **gen_kwargs):
       if isinstance(result, MultiBinPackerEnv):
           env_obj = result
           break
   used_packers = env_obj.used_packers
   for bin_idx, packer in enumerate(used_packers):
       for split in packer.splits:
           bins[bin_idx].append((split.x, split.y, split.z, split.width, split.height, split.depth))
   ```
   Esto es más robusto: reconstruye la asignación final leyendo directamente los objetos `SpacePartitioner` guardados en `env.used_packers`, en vez de confiar en que el orden de `bin_id` emitido por el generador coincida con el índice físico (evita errores si el orden de cierre de bins no es estrictamente secuencial).
3. **Celda 16 (`Gráfica 3D`)**: visualización interactiva con **Plotly** (`go.Mesh3d` + `go.Scatter3d`, renderer forzado a `"browser"`) del bin con más carga (`used_bins_indices[29]`, típicamente uno de los pallets M-1), alternativa a los gráficos estáticos de matplotlib usados en el resto del notebook.
4. **Celdas 18-22 (`Pruebas`)**: repetición del mismo bloque con `PATH='./input_int_plus4000.txt'` — otra calibración, esta vez con la carga "peor caso ampliada".
5. **Celdas 23-25**: repetición adicional (aparentemente sobre `input_best.txt`/`input_best_plus1000.txt`, seguidas de la gráfica Plotly) — última calibración antes de pasar a los vuelos reales.
6. **Celda 26 (markdown, `### Resultados Vuelos`) en adelante**: **el resultado final del proyecto**. Por cada uno de los 14 vuelos (`5Y-3462`, `5Y-3570`, `5Y-3570A`, `5Y-3572`, `5Y-3580`, `5Y-3580A`, `5Y-3582`, `5Y-3586`, `5Y-4570`, `5Y-6572`, `5Y-6576`, `5Y-7688`, `5Y-7734`, `5Y-8780`), una celda de código autocontenida (misma plantilla que la celda 14) que:
   - fija `PATH = './input_int_5Y-XXXX_plus2000.txt'` (el manifiesto real de ese vuelo),
   - corre `deeppack3d(METHOD, K=7, data='file', path=PATH, n_iterations=1)` sobre los 65 bins reales,
   - imprime, por cada uno de los 65 bins con carga: utilización %, volumen usado, volumen del contenedor, tamaño real y número de ítems,
   - grafica la utilización de las 65 posiciones (barras) y el contenido 3D del primer bin cargado,
   - imprime el total de cajas colocadas por bin y en todo el avión, el desglose de tamaños por bin, un listado de ejemplo de las primeras 20 cajas del bin 0, y finalmente compara contra `np.loadtxt(PATH)` para reportar **cuántas cajas del manifiesto quedaron sin embarcar** (`no_empacadas = total_items_file - total_packed`).

   Ejemplo concreto capturado en `Heuristica - BL.ipynb`, vuelo `5Y-3462` (método `bl`, `K=7`): de **12 537** cajas en el manifiesto, se generó un plan de estiba con utilización entre **0.86 y 0.88** en los 30 pallets M-1 (bins 0-29), cayendo a ~0.45-0.50 en las 23 franjas M-1H superiores (bins 30-52, más pequeñas y con formas menos "amigables" para las cajas del catálogo) y valores intermedios en P6P/LD-1/bulk (bins 53-64).

En conjunto, estos tres notebooks son el entregable operativo: **para cada vuelo y cada heurística, un plan de estiba caja-por-caja-y-posición**, más las métricas de utilización que permiten comparar `bl` vs `baf` vs `blsf` para decidir cuál conviene usar en la operación real.

---

## 6. Flujo de ejecución end-to-end (paso a paso)

Para una corrida típica de producción, p. ej. `Heuristica - BL.ipynb`, vuelo `5Y-3462`, método `bl`:

1. El notebook llama `deeppack3d('bl', 7, data='file', path='./input_int_5Y-3462_plus2000.txt', verbose=0, n_iterations=1)` (`deeppack3d.py:65`).
2. `deeppack3d()` construye la lista de 65 `bin_sizes` (líneas 71-76) y crea `MultiBinPackerEnv(n_bins=65, max_bins=65, k=7, bin_sizes=bin_sizes, ...)` (línea 86).
3. Sustituye el conveyor por `FileConveyor(k=7, path=...)` (línea 97), que carga perezosamente las 12 537 cajas del manifiesto (`conveyor.py:38-50`).
4. Como `method != 'rl'`, crea `HeuristicAgent(bottom_left, env, ...)` (`deeppack3d.py:150`, heurística de `agent.py:349`).
5. `agent.run(n_iterations=1, ...)` (`agent.py:296`) entra en el bucle principal:
   - `state = env.reset()` → `env.state()` (`env.py:123`) calcula `items` (7 cajas visibles), `h_maps` (65 mapas de altura, uno por bin) y `actions` (todas las colocaciones factibles: ítem × bin × rotación × posición).
   - `action = bottom_left(actions)` (`agent.py:349`) elige la colocación con menor `(y+h, x+w, z+d)`.
   - `env.step(action)` (`env.py:193`) coloca el cuboide en el `SpacePartitioner` correspondiente (`SpacePartitioner.add`, línea 96), actualiza `height_map`, recalcula `free_splits`, calcula la recompensa, y si el bin activo se agota, lo mueve a `used_packers` y abre el siguiente tamaño de `bin_sizes` (o termina si ya se usaron los 65).
   - Cada colocación se `yield`ea hacia `deeppack3d()`, que a su vez la `yield`ea hacia el notebook.
6. Cuando el conveyor se queda sin cajas o se agotan los 65 bins, `agent.run` termina el episodio (`yield None`) y `deeppack3d()` hace el **`yield env`** final (línea 165).
7. El notebook detecta `isinstance(result, MultiBinPackerEnv)`, guarda `env_obj`, y reconstruye `bins[0..64]` leyendo `env_obj.used_packers[i].splits`.
8. Se calculan métricas (`bin_utilization_real`), se grafican y se imprime el resumen (cajas colocadas, cajas sin embarcar, desglose de tamaños).

---

## 7. Parámetros configurables (referencia rápida)

| Parámetro | Dónde | Significado | Valor usado en producción |
|---|---|---|---|
| `method` | CLI / `deeppack3d()` | `'bl'`, `'baf'`, `'bssf'`, `'blsf'` (heurística) o `'rl'` | `'bl'`, `'baf'`, `'blsf'` (nunca `'bssf'` ni `'rl'` en producción final) |
| `lookahead` (`k`) | CLI / `deeppack3d()` / `env.k` | Nº de cajas futuras visibles simultáneamente para elegir la mejor colocación | `7` en los notebooks de vuelo (se probó `3,4,5,25,30,40,50,90` en calibración) |
| `data` | `deeppack3d()` | `'generated'` (sintético), `'input'` (teclado), `'file'` (manifiesto real) | `'file'` |
| `path` | `deeppack3d()` | Ruta al manifiesto `w h d` por línea | `input_int_5Y-XXXX_plus2000.txt` |
| `bin_sizes` | `MultiBinPackerEnv` | Lista ordenada de 65 tamaños `(W,H,D)` reales del 747-400F | Fija, hardcodeada (§4.2) |
| `replace` | `MultiBinPackerEnv` | `'all'` (repone todos los bins agotados a la vez) o `'max'` (uno a uno) | `'all'` (por defecto en `deeppack3d.py`) |
| `use_rotate` | `MultiBinPackerEnv` | Permite rotar cajas en los 3 ejes | `True` |
| `use_skip` | `MultiBinPackerEnv` | Permite saltarse ítems del lookahead que no caben | `True` |
| `n_iterations` | `deeppack3d()` | Episodios a correr (con `data='file'`, normalmente `1`: un vuelo = un episodio) | `1` |
| `verbose` | todos | `0` silencioso, `1` resumen por bin, `2` detalle paso a paso | `0` o `1` |
| `train` / `batch_size` | `deeppack3d()` / `Agent` | Solo relevantes para el camino RL (no usado en producción) | `False` |

---

## 8. Artefactos de salida

- `models/k=5.h5`, `models/k=10.h5`: pesos de la red Q entrenada para lookahead 5 y 10 (camino RL — no usados por los notebooks de producción por vuelo, que solo usan heurísticas).
- `outputs/`: imágenes `{episodio}_{paso}_{bin}.jpg` generadas por `Agent`/`HeuristicAgent` cuando `visualize=True` (`agent.py:253-255`, `333-335`), muestran el render 3D del bin tras cada colocación — son las imágenes referenciadas en el `README.md`.
- `dist/DeepPack3D-0.1.0-py3-none-any.whl`, `dist/deeppack3d-0.1.0.tar.gz`, `DeepPack3D.egg-info/`: artefactos de empaquetado estándar de `setuptools` (generados por `python setup.py sdist bdist_wheel`, ver `README.md`), permiten instalar el paquete como librería (`pip install`) en vez de ejecutarlo desde el código fuente.
- CSVs `bins_out/*.csv`, `pallet_*.csv`, `bin_real_*.csv`: exportaciones puntuales generadas por las funciones `export_bin_csv`/`export_csv` de `Testing_MIT.ipynb` (columnas `w,h,d,x,y,z`) — parte de los experimentos, no se generan en los notebooks de producción final (que solo imprimen y grafican).

---

## 9. Observaciones, inconsistencias y deuda técnica detectada

Estas notas no son errores "bloqueantes" (los notebooks corren y producen resultados), pero conviene tenerlas presentes si se va a seguir manteniendo o confiar en los números:

1. **Rotulado cruzado en `Heuristicas.ipynb`**: la celda markdown dice `Best_short_side_fit` pero el código de la celda siguiente fija `METHOD='baf'`; la celda markdown `Best_area_fit` va seguida de código con `METHOD='bssf'`. Es decir, las etiquetas de esas dos secciones están intercambiadas respecto al código real. No afecta a los notebooks de producción por vuelo (`Heuristica - BL/BAF/BLSF.ipynb`), donde el nombre del archivo sí coincide con el `METHOD` usado.
2. **Tamaño del bin "bulk" inconsistente en un punto**: `deeppack3d.py:76` y las tres `Heuristica - *.ipynb` usan `(20, 20, 18)`; la primera celda de `Heuristicas.ipynb` usa `(20, 20, 17)`. Diferencia menor (1 unidad de profundidad) pero es una lista copiada a mano en cada notebook en vez de importada desde un único lugar, así que cualquier corrección futura debe replicarse manualmente en cada copia.
3. **Normalización potencialmente incorrecta en el camino RL** (`agent.py:107`, `Agent.Q_inputs`): usa `self.env.size` (el tamaño de referencia global pasado a `MultiBinPackerEnv`) para normalizar la altura, en vez de `self.env.packers[j].size` (el tamaño real del bin `j`, que sí se usa correctamente en `env.py` para `p_map`/`i_map`). Con los 65 bins heterogéneos del 747, esto hace que la señal de entrada a la red Q no esté escalada de forma consistente entre bins de distinto tamaño — es una causa plausible de que el camino RL no llegara a resultados fiables y se abandonara en favor de las heurísticas puras para el entregable final.
4. **Código muerto**: `binpacker.py` (clase `BinPacker`, función `first_fit`) no lo importa ni usa ningún otro archivo del proyecto; toda la lógica de empaquetado real vive en `SpacePartitioner.py`. Es un remanente del paquete original.
5. **Prototipos abandonados sin rastro de código fuente**: `env_multisize_747_400.py` y `pack_with_pallet_plan.py` solo existen como bytecode compilado (`__pycache__/*.pyc`) — sus fuentes vivían fuera de esta carpeta (en `SIMPAC-2024-311-main`, según las rutas `os.chdir(...)` de `Testing_MIT.ipynb`) y ya no están disponibles aquí; su funcionalidad fue absorbida por `env.py` (`bin_sizes`) y por el uso directo de `deeppack3d()` respectivamente.
6. **Definición de `bin_sizes` duplicada ~20 veces**: la misma lista de 65 tuplas aparece copiada y pegada en `deeppack3d.py` y en (casi) cada celda de los cuatro notebooks de heurísticas. Cualquier cambio en la configuración real del avión requiere editar todas las copias a mano.
7. **Notebooks como bitácora, no como pipeline reproducible**: `Testing_MIT.ipynb` en particular contiene `os.chdir(...)` a rutas absolutas de la máquina/usuario original y celdas con *monkeypatching* en caliente sobre clases ya importadas — refleja un proceso iterativo de depuración, pero no es ejecutable "de un tirón" en otra máquina sin ajustar rutas.
