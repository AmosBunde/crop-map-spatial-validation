-- Training fields that sit within %(buffer_m)s metres of any test field.
-- These are dropped from training so the model cannot learn from the test area's
-- immediate neighbours. The GiST index on centroid makes ST_DWithin an index scan.
SELECT DISTINCT tr.tile, tr.field_id
FROM fields tr
JOIN fields te
  ON ST_DWithin(tr.centroid, te.centroid, %(buffer_m)s)
WHERE (tr.tile, tr.field_id) IN (SELECT * FROM unnest(%(train_tiles)s::text[], %(train_ids)s::int[]))
  AND (te.tile, te.field_id) IN (SELECT * FROM unnest(%(test_tiles)s::text[],  %(test_ids)s::int[]));
