/* Charts for the spatial-validation study. Plain D3 v7, no build step.
   CropViz.mount({ study, fields, el: { tiles, splits, buffer, map, variogram, table } })
   Colours and type come from CSS custom properties on .viz-root, so the same
   script runs on the project page and inside another site's stylesheet. */
(function (global) {
  'use strict';
  const d3 = global.d3;
  const SPLITS = [
    { key: 'random', name: 'Random fields', note: 'a random 20% of fields held out' },
    { key: 'block', name: '2 km blocks', note: 'whole 2 km squares held out' },
    { key: 'tile', name: 'Whole tile', note: 'one of four image tiles held out' }
  ];
  const MODELS = [
    { key: 'rf', name: 'Random forest', lower: 'random forest', cls: 'a' },
    { key: 'tcn', name: 'Temporal CNN', lower: 'temporal CNN', cls: 'b' }
  ];
  const f2 = d3.format('.2f');
  const f3 = d3.format('.3f');
  const pct = d3.format('.0%');

  function tooltip(root) {
    const tip = d3.select(root).append('div').attr('class', 'viz-tip').attr('role', 'status');
    return {
      show(html, event) {
        tip.html(html).classed('on', true);
        const box = root.getBoundingClientRect();
        const w = tip.node().offsetWidth, h = tip.node().offsetHeight;
        let x = event.clientX - box.left + 14, y = event.clientY - box.top - h - 12;
        if (x + w > box.width) x = event.clientX - box.left - w - 14;
        if (y < 0) y = event.clientY - box.top + 16;
        tip.style('left', x + 'px').style('top', y + 'px');
      },
      hide() { tip.classed('on', false); }
    };
  }

  function frame(el, height, margin) {
    const width = Math.max(260, el.clientWidth);
    d3.select(el).selectAll('svg').remove();
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${width} ${height}`).attr('width', '100%').attr('role', 'img');
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
    return { svg, g, w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  }

  function legend(el, items) {
    d3.select(el).selectAll('.viz-legend').remove();
    const box = d3.select(el).insert('div', ':first-child').attr('class', 'viz-legend');
    items.forEach((it) => {
      const row = box.append('span').attr('class', 'viz-key');
      row.append('span').attr('class', 'viz-swatch ' + it.cls);
      row.append('span').text(it.name);
    });
  }

  /* ---- headline tiles ---- */
  function tiles(el, study) {
    const hm = study.headline_model || 'rf';
    const rf = Object.fromEntries(study.main.filter((r) => r.model === hm).map((r) => [r.split, r]));
    const gap = study.gaps.find((g) => g.model === hm && g.comparison === 'random - tile');
    const root = d3.select(el).html('');
    const add = (value, label, sub) => {
      const t = root.append('div').attr('class', 'viz-stat');
      t.append('div').attr('class', 'viz-stat-num').text(value);
      t.append('div').attr('class', 'viz-stat-label').text(label);
      if (sub) t.append('div').attr('class', 'viz-stat-sub').text(sub);
    };
    SPLITS.forEach((s) => add(f2(rf[s.key].macro_f1), 'macro-F1, ' + s.name.toLowerCase(), s.note));
    if (gap) add((gap.macro_f1_gap >= 0 ? '+' : '') + f2(gap.macro_f1_gap), 'overstated by the random split',
      `95% interval ${f2(gap.ci95[0])} to ${f2(gap.ci95[1])}, against a held-out tile`);
  }

  /* ---- same models, three ways of testing ---- */
  function splits(el, study) {
    const narrow = el.clientWidth < 560;
    const m = { top: 8, right: 44, bottom: 40, left: narrow ? 104 : 150 };
    const { g, w, h } = frame(el, 250, m);
    const rows = study.main.filter((r) => r.model !== 'majority');
    const base = Object.fromEntries(study.main.filter((r) => r.model === 'majority').map((r) => [r.split, r.macro_f1]));
    const x = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.macro_f1_ci95[1]) * 1.08]).nice().range([0, w]);
    const y = d3.scaleBand().domain(SPLITS.map((s) => s.key)).range([0, h]).padding(0.25);
    const tip = tooltip(el);

    g.append('g').attr('class', 'viz-grid').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(narrow ? 3 : 6).tickSize(-h).tickFormat(f2)).call((a) => a.select('.domain').remove());
    g.append('text').attr('class', 'viz-axis-title').attr('x', w / 2).attr('y', h + 34).attr('text-anchor', 'middle').text('macro-F1 (higher is better)');
    SPLITS.forEach((s) => {
      const cy = y(s.key) + y.bandwidth() / 2;
      g.append('text').attr('class', 'viz-cat').attr('x', -12).attr('y', cy - 2).attr('text-anchor', 'end').text(s.name);
      g.append('text').attr('class', 'viz-cat-note').attr('x', -12).attr('y', cy + 14).attr('text-anchor', 'end').text(s.key === 'random' ? 'the usual test' : s.key === 'block' ? 'nearby, unseen' : 'a new district');
      if (base[s.key] != null) {
        g.append('line').attr('class', 'viz-base').attr('x1', x(base[s.key])).attr('x2', x(base[s.key])).attr('y1', y(s.key) + 2).attr('y2', y(s.key) + y.bandwidth() - 2);
      }
    });
    const firstBase = base.random;
    if (firstBase != null && !narrow) g.append('text').attr('class', 'viz-note').attr('x', x(firstBase) + 6).attr('y', y('random') + 8).text('always guess maize');

    MODELS.forEach((mod, i) => {
      const off = (i === 0 ? -1 : 1) * 9;
      const data = rows.filter((r) => r.model === mod.key);
      const sel = g.selectAll('.mark-' + mod.key).data(data).join('g').attr('class', 'mark-' + mod.key)
        .attr('transform', (d) => `translate(0,${y(d.split) + y.bandwidth() / 2 + off})`);
      sel.append('line').attr('class', 'viz-ci ' + mod.cls).attr('x1', (d) => x(d.macro_f1_ci95[0])).attr('x2', (d) => x(d.macro_f1_ci95[1]));
      sel.append('circle').attr('class', 'viz-dot ' + mod.cls).attr('r', 5.5).attr('cx', (d) => x(d.macro_f1));
      sel.append('text').attr('class', 'viz-val').attr('x', (d) => x(d.macro_f1_ci95[1]) + 8).attr('y', 4).text((d) => f2(d.macro_f1));
      sel.append('rect').attr('class', 'viz-hit').attr('x', 0).attr('y', -9).attr('width', w).attr('height', 18)
        .on('mousemove', (ev, d) => tip.show(`<b>${mod.name}</b><br>${SPLITS.find((s) => s.key === d.split).name}<br>macro-F1 ${f3(d.macro_f1)}<br>95% interval ${f3(d.macro_f1_ci95[0])} to ${f3(d.macro_f1_ci95[1])}<br>accuracy ${pct(d.accuracy)}`, ev))
        .on('mouseleave', tip.hide);
    });
    legend(el, MODELS);
  }

  /* ---- how far must the test fields be? ---- */
  function buffer(el, study) {
    const narrow = el.clientWidth < 560;
    const m = { top: 14, right: narrow ? 40 : 150, bottom: 44, left: 44 };
    const { g, w, h } = frame(el, 320, m);
    const data = study.buffer;
    const hm = study.headline_model || 'rf';
    const hmName = MODELS.find((mm) => mm.key === hm).lower;
    const ref = study.main.find((r) => r.model === hm && r.split === 'random');
    const hi = Math.max(d3.max(data, (d) => d.macro_f1_ci95[1]), ref ? ref.macro_f1 : 0);
    const x = d3.scaleLinear().domain([0, d3.max(data, (d) => d.buffer_m)]).range([16, w]);
    const y = d3.scaleLinear().domain([0, hi * 1.1]).nice().range([h, 0]);
    const tip = tooltip(el);

    g.append('g').attr('class', 'viz-grid').call(d3.axisLeft(y).ticks(5).tickSize(-w).tickFormat(f2)).call((a) => a.select('.domain').remove());
    g.append('g').attr('class', 'viz-axis').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x).tickValues([...new Set(data.map((d) => d.buffer_m))].filter((v) => !narrow || v === 0 || v >= 1000)).tickFormat((v) => (v === 0 ? '0' : v >= 1000 ? v / 1000 + ' km' : v + ' m')).tickSize(4))
      .call((a) => a.select('.domain').remove());
    g.append('text').attr('class', 'viz-axis-title').attr('x', w / 2).attr('y', h + 38).attr('text-anchor', 'middle').text('gap enforced between training and test fields');
    if (ref) {
      g.append('line').attr('class', 'viz-ref').attr('x1', 0).attr('x2', w).attr('y1', y(ref.macro_f1)).attr('y2', y(ref.macro_f1));
      g.append('text').attr('class', 'viz-note').attr('x', 4).attr('y', y(ref.macro_f1) - 6).text('random split, ' + hmName + ': ' + f2(ref.macro_f1));
    }
    const labelY = [];
    /* Control: the same shrinking training set, but with no gap enforced. If the
       buffered line falls below this one, distance is doing the damage, not size. */
    const ctrlAll = study.buffer_control || [];
    MODELS.forEach((mod) => {
      const zero = data.find((d) => d.model === mod.key && d.buffer_m === 0);
      const c = ctrlAll.filter((d) => d.model === mod.key).sort((a, b) => a.buffer_m - b.buffer_m);
      if (!zero || !c.length) return;
      g.append('path').datum([zero].concat(c)).attr('class', 'viz-line viz-control ' + mod.cls)
        .attr('d', d3.line().x((d) => x(d.buffer_m)).y((d) => y(d.macro_f1)));
    });
    MODELS.forEach((mod) => {
      const s = data.filter((d) => d.model === mod.key).sort((a, b) => a.buffer_m - b.buffer_m);
      if (!s.length) return;
      g.append('path').datum(s).attr('class', 'viz-band ' + mod.cls)
        .attr('d', d3.area().x((d) => x(d.buffer_m)).y0((d) => y(d.macro_f1_ci95[0])).y1((d) => y(d.macro_f1_ci95[1])));
      g.append('path').datum(s).attr('class', 'viz-line ' + mod.cls).attr('d', d3.line().x((d) => x(d.buffer_m)).y((d) => y(d.macro_f1)));
      g.selectAll('.pt-' + mod.key).data(s).join('circle').attr('class', 'viz-dot ' + mod.cls).attr('r', 4.5).attr('cx', (d) => x(d.buffer_m)).attr('cy', (d) => y(d.macro_f1));
      const last = s[s.length - 1];
      let ly = y(last.macro_f1) + 4;
      labelY.forEach((p) => { if (Math.abs(p - ly) < 15) ly = p + (ly >= p ? 15 : -15); });
      labelY.push(ly);
      g.append('text').attr('class', 'viz-val').attr('x', w + 10).attr('y', ly).text(narrow ? f2(last.macro_f1) : `${mod.name} ${f2(last.macro_f1)}`);
    });
    const xs = [...new Set(data.map((d) => d.buffer_m))].sort((a, b) => a - b);
    const cross = g.append('line').attr('class', 'viz-cross').attr('y1', 0).attr('y2', h).attr('opacity', 0);
    g.append('rect').attr('class', 'viz-hit').attr('width', w).attr('height', h)
      .on('mousemove', (ev) => {
        const px = d3.pointer(ev)[0];
        const b = xs.reduce((a, c) => (Math.abs(x(c) - px) < Math.abs(x(a) - px) ? c : a));
        cross.attr('x1', x(b)).attr('x2', x(b)).attr('opacity', 1);
        const lines = MODELS.map((mod) => {
          const d = data.find((r) => r.model === mod.key && r.buffer_m === b);
          return d ? `<span class="viz-swatch ${mod.cls}"></span>${mod.name}: <b>${f3(d.macro_f1)}</b> (${f3(d.macro_f1_ci95[0])} to ${f3(d.macro_f1_ci95[1])})` : '';
        }).join('<br>');
        const any = data.find((r) => r.buffer_m === b);
        const extra = MODELS.map((mod) => { const cc = ctrlAll.find((r) => r.model === mod.key && r.buffer_m === b); return cc ? `<br>${mod.name}, same training size with no gap: <b>${f3(cc.macro_f1)}</b>` : ''; }).join('');
        tip.show(`<b>${b === 0 ? 'No gap' : b + ' m gap'}</b><br>${lines}${extra}<br>mean training fields: ${Math.round(any.mean_train_size)}`, ev);
      })
      .on('mouseleave', () => { cross.attr('opacity', 0); tip.hide(); });
    legend(el, MODELS);
  }

  /* ---- where the errors are ---- */
  function map(el, study, fields) {
    const host = d3.select(el);
    host.selectAll('*').remove();
    const controls = host.append('div').attr('class', 'viz-filter').attr('role', 'group').attr('aria-label', 'Test design shown on the map');
    const status = host.append('p').attr('class', 'viz-map-status').attr('aria-live', 'polite');
    const stage = host.append('div').attr('class', 'viz-map-stage');
    const width = Math.max(260, el.clientWidth);
    const height = Math.round(Math.min(760, width * 1.18));
    const svg = stage.append('svg').attr('viewBox', `0 0 ${width} ${height}`).attr('width', '100%').attr('role', 'img')
      .attr('aria-label', 'Map of labelled fields in Busia County, coloured by whether the model classified each correctly');
    /* Rings are projected point by point and drawn as plain paths: d3-geo reads
       polygon winding spherically, and a 40 km square needs no great-circle care. */
    const corners = Object.values(study.tiles).flat();
    const proj = d3.geoMercator().fitExtent([[10, 10], [width - 10, height - 10]], { type: 'MultiPoint', coordinates: corners });
    const ring = (r) => 'M' + r.map((c) => proj(c).map((v) => v.toFixed(1)).join(',')).join('L') + 'Z';
    const world = svg.append('g');
    if (study.blocks) {
      world.append('g').selectAll('path').data(study.blocks).join('path').attr('class', 'viz-block').attr('d', (b) => ring(b.ring));
    }
    const tileList = Object.entries(study.tiles).map(([t, r]) => ({ t, r }));
    world.append('g').selectAll('path').data(tileList).join('path').attr('class', 'viz-tile').attr('d', (d) => ring(d.r));
    const tileLabels = world.append('g').selectAll('text').data(tileList).join('text').attr('class', 'viz-tile-label')
      .attr('x', (d) => d3.min(d.r, (c) => proj(c)[0]) + 8).attr('y', (d) => d3.min(d.r, (c) => proj(c)[1]) + 18).text((d) => 'tile ' + d.t);

    const n = fields.lon.length;
    const pts = d3.range(n).map((i) => { const p = proj([fields.lon[i], fields.lat[i]]); return { i, x: p[0], y: p[1] }; });
    const layer = world.append('g');
    const dots = layer.selectAll('circle').data(pts).join('circle').attr('r', 2.6).attr('cx', (d) => d.x).attr('cy', (d) => d.y);
    const delaunay = d3.Delaunay.from(pts, (d) => d.x, (d) => d.y);
    const focus = world.append('circle').attr('class', 'viz-focus').attr('r', 6).attr('opacity', 0);
    const tip = tooltip(stage.node());
    let current = 'tile';
    let k = 1, tf = d3.zoomIdentity;
    /* The wheel zooms only with Ctrl or Cmd held, so the page still scrolls past the map. */
    const zoom = d3.zoom().filter((ev) => (ev.type === 'wheel' ? ev.ctrlKey || ev.metaKey : !ev.button)).scaleExtent([1, 60]).translateExtent([[0, 0], [width, height]]).on('zoom', (ev) => {
      tf = ev.transform; k = tf.k;
      world.attr('transform', tf);
      dots.attr('r', 2.6 / Math.sqrt(k)).attr('stroke-width', 0.6 / k);
      focus.attr('r', 6 / Math.sqrt(k)).attr('stroke-width', 2 / k);
      world.selectAll('.viz-tile').attr('stroke-width', 1.5 / k);
      world.selectAll('.viz-block').attr('stroke-width', 1 / k);
      tileLabels.attr('font-size', 12.5 / k);
      reset.property('disabled', k === 1);
    });
    svg.call(zoom).on('dblclick.zoom', null);
    controls.append('button').attr('type', 'button').attr('class', 'viz-chip viz-zoom').attr('aria-label', 'Zoom in').text('+')
      .on('click', () => svg.transition().duration(200).call(zoom.scaleBy, 2));
    controls.append('button').attr('type', 'button').attr('class', 'viz-chip').attr('aria-label', 'Zoom out').text('\u2212')
      .on('click', () => svg.transition().duration(200).call(zoom.scaleBy, 0.5));
    const reset = controls.append('button').attr('type', 'button').attr('class', 'viz-chip viz-reset').text('Reset zoom').property('disabled', true)
      .on('click', () => svg.call(zoom.transform, d3.zoomIdentity));
    host.append('p').attr('class', 'viz-hint').text('Labels were collected along walking routes, so fields sit in tight clusters. Use + to zoom in, or hold Ctrl and scroll; drag to move.');

    function draw() {
      const ok = fields['ok_' + current];
      dots.attr('class', (d) => 'viz-pt ' + (ok[d.i] ? 'a' : 'b'));
      layer.selectAll('circle').filter((d) => !ok[d.i]).raise();
      const right = d3.sum(ok);
      status.text(`${SPLITS.find((s) => s.key === current).name}: ${right.toLocaleString()} of ${n.toLocaleString()} fields classified correctly (${pct(right / n)}) by the ${MODELS.find((mm) => mm.key === (study.headline_model || 'rf')).lower}.`);
      controls.selectAll('button.viz-split').attr('aria-pressed', (d) => String(d.key === current));
    }
    controls.selectAll('button.viz-split').data(SPLITS).join((e) => e.insert('button', '.viz-zoom')).attr('type', 'button').attr('class', 'viz-chip viz-split').text((d) => d.name)
      .on('click', (ev, d) => { current = d.key; draw(); });
    svg.on('mousemove', (ev) => {
      const [sx, sy] = d3.pointer(ev);
      const [mx, my] = tf.invert([sx, sy]);
      const k = delaunay.find(mx, my), p = pts[k];
      if (Math.hypot(p.x - mx, p.y - my) * tf.k > 24) { focus.attr('opacity', 0); tip.hide(); return; }
      focus.attr('cx', p.x).attr('cy', p.y).attr('opacity', 1);
      const ok = fields['ok_' + current][k];
      const extra = current === 'tile' && !ok ? `<br>predicted: ${fields.classes[fields.pred_tile[k]]}` : '';
      tip.show(`<b>${fields.classes[fields.label[k]]}</b><br>${ok ? 'classified correctly' : 'misclassified'}${extra}<br>tile ${fields.tile[k]}, ${fields.pixels[k]} pixel${fields.pixels[k] === 1 ? '' : 's'}`, ev);
    }).on('mouseleave', () => { focus.attr('opacity', 0); tip.hide(); });
    draw();
    legend(el, [{ name: 'Classified correctly', cls: 'a' }, { name: 'Misclassified', cls: 'b' }]);
  }

  /* ---- how far does similarity reach? ---- */
  function variogram(el, study) {
    const v = study.variogram;
    const m = { top: 14, right: 20, bottom: 44, left: 56 };
    const { g, w, h } = frame(el, 280, m);
    const x = d3.scaleLinear().domain([0, d3.max(v.bins, (d) => d.lag_km)]).nice().range([0, w]);
    const y = d3.scaleLinear().domain([0, Math.max(v.sill, d3.max(v.bins, (d) => d.semivariance)) * 1.15]).nice().range([h, 0]);
    const tip = tooltip(el);
    g.append('g').attr('class', 'viz-grid').call(d3.axisLeft(y).ticks(5).tickSize(-w).tickFormat(d3.format('.3f'))).call((a) => a.select('.domain').remove());
    g.append('g').attr('class', 'viz-axis').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(6).tickFormat((d) => d + ' km').tickSize(4)).call((a) => a.select('.domain').remove());
    g.append('text').attr('class', 'viz-axis-title').attr('x', w / 2).attr('y', h + 38).attr('text-anchor', 'middle').text('distance between two fields');
    g.append('line').attr('class', 'viz-ref').attr('x1', 0).attr('x2', w).attr('y1', y(v.sill)).attr('y2', y(v.sill));
    g.append('text').attr('class', 'viz-note').attr('x', w).attr('y', y(v.sill) - 6).attr('text-anchor', 'end').text('any two fields, whatever the distance');
    g.append('path').datum(v.bins).attr('class', 'viz-line a').attr('d', d3.line().x((d) => x(d.lag_km)).y((d) => y(d.semivariance)));
    g.selectAll('circle').data(v.bins).join('circle').attr('class', 'viz-dot a').attr('r', 4.5).attr('cx', (d) => x(d.lag_km)).attr('cy', (d) => y(d.semivariance));
    const cross = g.append('line').attr('class', 'viz-cross').attr('y1', 0).attr('y2', h).attr('opacity', 0);
    g.append('rect').attr('class', 'viz-hit').attr('width', w).attr('height', h)
      .on('mousemove', (ev) => {
        const px = d3.pointer(ev)[0];
        const d = v.bins.reduce((a, c) => (Math.abs(x(c.lag_km) - px) < Math.abs(x(a.lag_km) - px) ? c : a));
        cross.attr('x1', x(d.lag_km)).attr('x2', x(d.lag_km)).attr('opacity', 1);
        tip.show(`<b>${d3.format('.1f')(d.lag_km)} km apart</b><br>NDVI semivariance ${d3.format('.4f')(d.semivariance)}<br>${pct(d.semivariance / v.sill)} of the overall variance<br>${d.pairs.toLocaleString()} field pairs`, ev);
      }).on('mouseleave', () => { cross.attr('opacity', 0); tip.hide(); });
  }

  /* ---- the numbers, as a table ---- */
  function table(el, study) {
    const root = d3.select(el).html('').append('div').attr('class', 'viz-table-wrap');
    const t = root.append('table').attr('class', 'viz-table');
    t.append('caption').text('Macro-F1 by test design and model, mean of five seeds, with 95% block-bootstrap intervals');
    const head = t.append('thead').append('tr');
    ['Test design', 'Model', 'Macro-F1', '95% interval', 'Accuracy', 'Calibration error'].forEach((h) => head.append('th').attr('scope', 'col').text(h));
    const body = t.append('tbody');
    study.main.forEach((r) => {
      const tr = body.append('tr');
      tr.append('th').attr('scope', 'row').text(SPLITS.find((s) => s.key === r.split).name);
      tr.append('td').text(r.model === 'majority' ? 'Majority class' : MODELS.find((m) => m.key === r.model).name);
      tr.append('td').text(f3(r.macro_f1));
      tr.append('td').text(`${f3(r.macro_f1_ci95[0])} to ${f3(r.macro_f1_ci95[1])}`);
      tr.append('td').text(pct(r.accuracy));
      tr.append('td').text(r.model === 'majority' ? 'n/a' : f3(r.ece));
    });
  }

  function mount(opts) {
    const { study, fields, el } = opts;
    const render = () => {
      if (el.tiles) tiles(el.tiles, study);
      if (el.splits) splits(el.splits, study);
      if (el.buffer) buffer(el.buffer, study);
      if (el.map) map(el.map, study, fields);
      if (el.variogram) variogram(el.variogram, study);
      if (el.table) table(el.table, study);
    };
    render();
    let timer = null, lastWidth = global.innerWidth;
    global.addEventListener('resize', () => {
      if (global.innerWidth === lastWidth) return;
      lastWidth = global.innerWidth;
      clearTimeout(timer);
      timer = setTimeout(render, 150);
    });
  }

  global.CropViz = { mount };
})(window);
