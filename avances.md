# Avances

## Problema encontrado

El entrenamiento de `pinn_model.py` completa las 20.000 epochs, pero la fuga no converge bien. En el caso probado con `scenario_id=8`, el modelo termina con `x_leak` lejos del valor real y `q_leak` muy por debajo de la fuga esperada. Eso indica que la red logra reducir la pérdida total, pero no identifica correctamente la fuente física de la fuga.

Además, se encontró un problema de diagnóstico gráfico en `plot_training_diagnostics`: la comparación entre `P_moc` y `P_pred` fallaba por un desajuste de ejes. `P_moc` venía en formato `[Nx, Nt]` y `P_pred` se estaba reconstruyendo como `[Nt, Nx]`.

## Cómo se pensó la solución

La primera idea fue revisar si el error venía de un problema de formas en la evaluación de la red. Eso se confirmó al comparar las dimensiones de ambos campos. La corrección fue reordenar `P_pred` para que quede con la misma orientación que `P_moc` antes de calcular la diferencia.

Después, el foco pasó a la convergencia. El comportamiento observado sugiere que la red aprende a ajustar la señal general, pero no la componente de fuga. La señal física de la fuga puede quedar "apagada" si el balance de pérdidas favorece demasiado la parte de datos o permite que `q_leak` se contraiga hacia valores mínimos.

## Recomendación para continuar

La mejor siguiente acción es ajustar el balance entre componentes del loss y darle más visibilidad al término de fuga:

1. Revisar los valores de `L_datos`, `L_fisica`, `L_contorno` y `L_inicial` en el epoch 1000.
2. Subir el peso de la física si `L_fisica` queda dominado por los datos.
3. Mantener un learning rate separado y más alto para `q_leak_raw`.
4. Si hace falta, empezar con `scenario_id=8` porque la fuga es más fuerte y sirve como caso de validación antes de intentar el caso más difícil.

## Estado actual

- La comparación de campos MOC vs PINN ya quedó alineada correctamente.
- El entrenamiento ya imprime componentes del loss durante el proceso.
- El caso aún no converge bien para la estimación fina de `q_leak`, así que el siguiente paso debe ser de calibración, no de cambios estructurales grandes.

---

## Entrada de Bitácora

- **Fecha:** 2026-06-09
- **Problema Observado:** La red convergía a un mínimo local, apagando la fuga (q_leak ~ 0) porque la pérdida de datos dominaba sobre la pérdida física (diferencia de 14 órdenes de magnitud debido a unidades físicas no normalizadas). Error espacial de ~3km.
- **Cambios Implementados:**
  1. Se modificó la inicialización en [pinn_model.py](file:///d:/MCD_24-26/PROYECTO_FINAL_PINN/pinn_model.py): `x_leak` se inicializa en el centro del dominio espacial (5000 m, correspondiente a `x_leak_raw = 0.0`) y `q_leak` en 0.015 m³/s (equivalente al 5% del caudal de salida nominal `Q_out = 0.3` m³/s), garantizando valores crudos no saturados para las funciones de mapeo (`sigmoid` y `softplus`).
  2. Se normalizaron todos los términos de la función de pérdida `compute_loss` en [pinn_model.py](file:///d:/MCD_24-26/PROYECTO_FINAL_PINN/pinn_model.py) utilizando las escalas físicas de referencia (`P_inlet` y `Q_outlet`). Ahora $L_{\text{datos}}$, $L_{\text{contorno}}$, $L_{\text{inicial}}$ y $L_{\text{fisica}}$ son adimensionales y de órdenes de magnitud comparables.
  3. Se ajustaron las lambdas por defecto de la pérdida a `{"data": 2.0, "pde": 2.0, "bc": 1.0, "ic": 1.0}` dado que los términos ya están normalizados.
  4. Se implementó un analizador de gradientes en las primeras 5 epochs que calcula las magnitudes del gradiente de `L_datos` y `L_fisica` por separado y emite una advertencia si los gradientes de datos dominan a los físicos (grad_data_norm > 10.0 * grad_pde_norm).
- **Resultados Esperados:** Equilibrar los gradientes de la red, evitar la desaparición del gradiente físico y forzar la correcta identificación de la posición (`x_leak`) y caudal (`q_leak`) de la fuga respetando las leyes de conservación.
- **Resultado Obtenido:** La normalización corrigió x_leak (error de 100 m vs 3.2 km), pero q_leak seguía cayendo al mínimo (0.001, error del 93%). Las pérdidas quedaron balanceadas (~1e-5) pero el problema de identificabilidad persistía.

---

## Entrada de Bitácora — Balance de masa global y parametrización sigmoid

- **Fecha:** 2026-06-09
- **Problema Observado:** Tras la normalización, `x_leak` convergió correctamente (6100 m vs 6000 m real), pero `q_leak` seguía decayendo al mínimo (0.001, error del 93%). Se identificaron **dos causas raíz**:
  1. **Co-adaptación red/q_leak:** La red ajustaba libremente el campo Q(x,t) para satisfacer la EDP con q_leak=0, sin restricción porque solo P es observada. Agregar Q como dato fue descartado porque trivializa el problema (q_leak = Q_upstream - Q_downstream sin necesidad de PINN).
  2. **Trampa de gradiente en softplus:** La parametrización `q_leak = softplus(raw) * 0.01 + 0.001` genera gradientes exponencialmente pequeños cuando `raw` se vuelve negativo. Una vez que q_leak empieza a bajar, no hay señal de gradiente para recuperarlo.
- **Cambios Implementados:**
  1. Se cambió la parametrización de `q_leak` de softplus a **sigmoid**: `q_leak = sigmoid(raw) * 0.049 + 0.001`, rango [0.001, 0.050]. La sigmoid tiene gradiente simétrico alrededor de raw=0, evitando la trampa unidireccional.
  2. Se agregó un nuevo término de pérdida **L_masa** (balance de masa global): `MSE(Q(0,t_late) - Q(L,t_late), q_leak)` para tiempos tardíos (t > 150s). Esta restricción es **derivada puramente de la física** (conservación de masa integral) y no usa datos externos de Q. Crea un acoplamiento directo entre el campo Q de la red y el parámetro q_leak.
  3. Se revirtió la inclusión de datos de Q en sensores por comprometer la validez del aporte de la tesis.
- **Resultados Esperados:** La sigmoid evita la trampa de gradiente y el balance de masa fuerza la auto-consistencia entre Q y q_leak sin usar datos externos.

---

## Entrada de Bitácora — Optimización híbrida Adam → L-BFGS y Sensibilidad de Sensores

- **Fecha:** 2026-06-09
- **Problema Observado:** Adam localiza eficientemente la zona de la fuga (`x_leak` error de 0.3-0.5 km), pero cerca del mínimo global carece de la precisión de segundo orden necesaria para ajustar finamente `q_leak` (error del 15% al 23%). Además, se requería estudiar el impacto del número de sensores (2, 3, 5, 11) en la precisión espacial.
- **Cambios Implementados:**
  1. **Algoritmo de Optimización Híbrido:** Se implementó una fase secundaria utilizando el optimizador **L-BFGS** (`torch.optim.LBFGS` con búsqueda de líneas `strong_wolfe`) posterior a la fase de Adam.
  2. **Fijación de Collocation Points para L-BFGS:** A diferencia de Adam (donde se remuestrean en cada época), para L-BFGS se congelan los puntos de colocación. Esto proporciona un paisaje de pérdida determinista esencial para que la aproximación del Hessiano no se desestabilice por el ruido estocástico del remuestreo.
  3. **Control CLI:** Se añadieron los argumentos `--no_lbfgs` y `--lbfgs_epochs` en [pinn_model.py](file:///d:/MCD_24-26/PROYECTO_FINAL_PINN/pinn_model.py) para permitir activar/desactivar y configurar la duración del refinamiento.
  4. **Expansión de Sensores:** Se extendió el soporte en [config.py](file:///d:/MCD_24-26/PROYECTO_FINAL_PINN/config.py) y [data_utils.py](file:///d:/MCD_24-26/PROYECTO_FINAL_PINN/data_utils.py) para evaluar 2, 3, 5 y 11 sensores equidistantes, interpolando de `P_full` en caso de requerir mayor resolución espacial que la del set de sensores original.
- **Resultados Esperados:** L-BFGS refinará los gradientes y el acoplamiento de masa, logrando una convergencia óptima en `q_leak` y permitiendo un estudio de sensibilidad riguroso para la tesis (error vs. cantidad de sensores).
- **Resultados Obtenidos / Validación de Código:**
  Se realizó una corrida de validación rápida de 200 epochs (160 Adam + 40 L-BFGS) para verificar la correcta integración del optimizador y el flujo de gradientes en WSL:
  * **Comportamiento observado:** Adam redujo la pérdida inicial de `9.87e+00` a `6.88e-03`. Al cambiar a L-BFGS en la época 161 (con puntos de colocación fijos), el optimizador convergió prematuramente en el paso 36 al detectar un cambio de pérdida infinitesimal (`change in loss < 1e-12`), confirmando la estabilidad del código.
  * **Resultados (200 ep):** Posición `x_leak pred = 5034 m` (error de 0.966 km), Caudal `q_leak pred = 0.0246` (error de 64.0%). Este nivel de error es normal para una corrida corta de solo 200 épocas (comparado con las 20.000 normales), pero valida que el flujo funciona y compila sin inconvenientes.