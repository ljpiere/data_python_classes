SELECT
    pais,
    clave_territorio,
    SUM(ingreso_total)::integer AS ingresos,
    SUM(costo_total)::integer  AS costos
FROM ventas_clean
GROUP BY
    pais,
    clave_territorio
ORDER BY
    ingresos DESC;