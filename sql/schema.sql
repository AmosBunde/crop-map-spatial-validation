-- One row per field. Geometry is stored in the recovered CRS (EPSG:32636, metres),
-- so distances and areas are metric without reprojection.
CREATE EXTENSION IF NOT EXISTS postgis;

DROP TABLE IF EXISTS fields CASCADE;
CREATE TABLE fields (
    tile        text    NOT NULL,
    field_id    integer NOT NULL,
    label       integer NOT NULL,          -- 0 = unlabelled
    n_pixels    integer NOT NULL,
    area_m2     double precision NOT NULL,
    geom        geometry(MultiPolygon, 32636) NOT NULL,
    centroid    geometry(Point, 32636) GENERATED ALWAYS AS (ST_Centroid(geom)) STORED,
    PRIMARY KEY (tile, field_id)
);
CREATE INDEX fields_geom_gix     ON fields USING gist (geom);
CREATE INDEX fields_centroid_gix ON fields USING gist (centroid);
CREATE INDEX fields_label_ix     ON fields (label);
