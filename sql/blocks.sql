-- Assign every labelled field to a square block on the map grid.
-- %(block_m)s is the block edge in metres. The same arithmetic as splits.block_ids.
SELECT tile, field_id,
       floor(ST_X(centroid) / %(block_m)s)::bigint AS bx,
       floor(ST_Y(centroid) / %(block_m)s)::bigint AS by
FROM fields
WHERE label > 0
ORDER BY tile, field_id;
