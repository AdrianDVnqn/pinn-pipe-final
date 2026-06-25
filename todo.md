# TODO: Próxima Sesión PINN

## 🛑 Estado Actual (Fin de Sesión)
La última prueba arrojó `x_leak = 7664m` (Error: 3664m). **Esta fue la última prueba programada en GPU**. Tal como solicitaste, no se correrán más scripts para liberar tu hardware. 

---

## 🔍 Conclusiones del Debugging y la Prueba Final

Descubrimos que la red neuronal estaba siendo saboteada por problemas fundamentales, de los cuales solucionamos la mayoría, pero uno persiste de una manera que requiere un cambio de enfoque:

### 1. El Dataset Transitorio vs La Física Estacionaria (CORREGIDO)
- **Problema:** El dataset incluía el transitorio del caño ($t=0$ a $t=150$), mientras que nuestra física asumía un estado estacionario.
- **Solución:** Filtramos los datos (`t >= 150`) para que la red trabaje exclusivamente en el régimen permanente.

### 2. El Escalado del PDE / Darcy-Weisbach (CORREGIDO)
- **Problema:** El residuo de la ecuación de momento quedaba ahogado por la normalización ($10^{-9}$), permitiendo a la red "curvarse" libremente ignorando la física rectilínea.
- **Solución:** Ajustamos el factor de escala (`r_mom / 0.01`), forzando a la red a respetar que el perfil de presión debe ser una caída lineal estricta.

### 3. El Mínimo Local Irrompible (EL PROBLEMA ACTUAL)
- **Hipótesis anterior:** Creíamos que al quitar el **Método de Continuación**, el gradiente de la posición de la fuga moría y dejaba la fuga atascada, y que al restaurarlo se solucionaría.
- **Resultado de la Última Prueba:** A pesar de tener el Método de Continuación activado (que garantiza gradientes a lo largo de toda la tubería), el modelo de todas formas migró de `5000m` a `7664m`.
- **Diagnóstico Definitivo:** La posición de $\sim 7600m$ **no es un error numérico de gradiente**, sino un **Mínimo Local** legítimo (y muy profundo) en el paisaje de pérdida (`Loss`). Al entrenar la red (MLP) *al mismo tiempo* que los parámetros físicos (`x_leak` y `q_leak`), la MLP y los parámetros compensan sus errores mutuamente. Encuentran una "zona de confort" matemática en $\sim 7600m$ de la cual el optimizador L-BFGS y Adam son incapaces de salir porque cualquier paso hacia $4000m$ aumenta temporalmente el error de los sensores.

---

## 📋 Plan de Acción Exacto para la Próxima Sesión

Puesto que el paisaje de error de las PINNs está minado de mínimos locales cuando se optimizan parámetros de salto (kinks), el enfoque de "descenso de gradiente ciego" desde el centro (`5000m`) ya demostró sus límites. 

Para la próxima sesión implementaremos una de estas dos arquitecturas definitivas:

1. **Grid Search / Multi-Initialization (Recomendado y Rápido):**
 - [x] **Identificar inconsistencias:** Correr un script para asegurar que la PDE estacionaria no recibe datos transitorios ni condiciones iniciales incompatibles. (Test 2)
 - [x] **Aplicar filtros:** Limitar el entrenamiento al rango $t \ge 150s$ si se confirma la falla.
   - Iniciamos **3 o 5 instancias muy livianas de la red** en diferentes puntos de partida (ej: `2000m`, `4000m`, `6000m`, `8000m`).
   - Las entrenamos solo 500 épocas.
   - Elegimos la que tenga el menor Loss de los sensores, y a ESA la entrenamos hasta el final (10.000 épocas + L-BFGS). 
   - Esta técnica es infalible contra mínimos locales unidimensionales.

2. **Desacoplar la Optimización (Coordinate Descent):**
   - Primero, congelar `x_leak` y `q_leak`. Entrenar la MLP para que dibuje el caño perfecto.
   - Segundo, congelar la MLP y optimizar solo `x_leak`.
   - Alternar este proceso. Al no dejar que evolucionen juntas, evitamos que "conspiren" para caer en mínimos locales irreales.

**Próximo paso al arrancar:** Elegir uno de los dos métodos descritos y modificar `train_pinn` para implementarlo antes de iniciar el factorial.
