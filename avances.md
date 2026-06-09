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