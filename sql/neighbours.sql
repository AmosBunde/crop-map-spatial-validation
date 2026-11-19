-- How clustered are the labels? For each labelled field, find its nearest labelled
-- neighbour (KNN on the GiST index) and record the distance and whether the two
-- share a crop. If neighbours agree far more often than chance, a random split
-- hands the model the answer for much of the test set.
SELECT f.tile, f.field_id, f.label,
       n.label AS neighbour_label,
       ST_Distance(f.centroid, n.centroid) AS distance_m
FROM fields f
CROSS JOIN LATERAL (
    SELECT g.label, g.centroid
    FROM fields g
    WHERE g.label > 0 AND (g.tile, g.field_id) <> (f.tile, f.field_id)
    ORDER BY g.centroid <-> f.centroid
    LIMIT 1
) n
WHERE f.label > 0;
