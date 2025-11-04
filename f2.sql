SELECT
    v.pais,
    v.clave_territorio,
    SUM(v.ingreso_total)::integer AS ingresos,
    SUM(v.costo_total)::integer AS costos,
    COALESCE(SUM(c.costo_campana ::integer), 0) AS costo_campana
FROM ventas_clean AS v
LEFT JOIN campanas AS c
  ON CAST(v.clave_territorio AS TEXT) = c.clave_territorio
GROUP BY
    v.pais,
    v.clave_territorio
ORDER BY ingresos DESC;