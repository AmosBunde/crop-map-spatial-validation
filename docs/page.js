Promise.all([fetch('data/study.json').then((r) => r.json()), fetch('data/fields.json').then((r) => r.json())])
  .then(([study, fields]) => {
    const el = {};
    ['tiles', 'splits', 'buffer', 'map', 'variogram', 'table'].forEach((id) => { el[id] = document.getElementById(id); });
    window.CropViz.mount({ study, fields, el });
  });
