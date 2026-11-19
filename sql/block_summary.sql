-- Per-block inventory: how many labelled fields, how many crops, and the block's
-- outline as geometry (for the map). Shows how unevenly the labels were collected.
SELECT floor(ST_X(centroid) / %(block_m)s)::bigint AS bx,
       floor(ST_Y(centroid) / %(block_m)s)::bigint AS by,
       count(*)                    AS n_fields,
       count(DISTINCT label)       AS n_classes,
       mode() WITHIN GROUP (ORDER BY label) AS majority_label,
       ST_AsText(ST_Transform(ST_MakeEnvelope(
           floor(ST_X(min(centroid)) / %(block_m)s) * %(block_m)s,
           floor(ST_Y(min(centroid)) / %(block_m)s) * %(block_m)s,
           floor(ST_X(min(centroid)) / %(block_m)s) * %(block_m)s + %(block_m)s,
           floor(ST_Y(min(centroid)) / %(block_m)s) * %(block_m)s + %(block_m)s,
           32636), 4326)) AS outline_wkt
FROM fields
WHERE label > 0
GROUP BY 1, 2
ORDER BY 1, 2;
