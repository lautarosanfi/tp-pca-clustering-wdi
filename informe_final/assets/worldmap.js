/*
 * worldmap.js — Mapa coroplético del score PC1 por país (2023).
 *
 * Diseño limpio y minimalista (estilo dashboard): fondo crema, sin marco, sin
 * costas ni océano, países sin dato en gris cálido neutro y una escala divergente
 * de la paleta del informe (terracota → crema → teal).
 *
 * Depende de:
 *   - window.Plotly         (cargado de forma diferida por report.js)
 *   - window.MAP_PC1_DATA   (definido en worldmap.data.js)
 *
 * API:  WorldMap.render(el, { interactive: false })  /  WorldMap.purge(el)
 */
(function () {
  "use strict";

  // Escala divergente centrada en 0 (PC1 = gradiente de desarrollo).
  var COLORSCALE = [
    [0.00, "#b23a26"], // terracota intenso  → menor desarrollo relativo
    [0.22, "#dd8a5a"], // tangerina
    [0.50, "#f1ede2"], // crema (centro, ~0)
    [0.78, "#4f9d95"], // teal medio
    [1.00, "#0d6d77"]  // teal profundo      → mayor desarrollo relativo
  ];

  var PAPER = "#f6f3ea";   // crema de fondo
  var NODATA = "#e4e0d4";  // relleno de países sin dato
  var INK = "#2a363b";

  function buildTrace(data, interactive) {
    return [{
      type: "choropleth",
      locationmode: "ISO-3",
      locations: data.map(function (d) { return d.iso; }),
      z: data.map(function (d) { return d.pc1; }),
      zmid: 0,
      colorscale: COLORSCALE,
      text: data.map(function (d) {
        return "<b>" + d.name + "</b><br>" +
          "Score PC1: " + d.pc1.toFixed(2) + "<br>" +
          "Ingreso: " + d.income + "<br>" +
          "Región: " + d.region;
      }),
      hovertemplate: interactive ? "%{text}<extra></extra>" : undefined,
      hoverinfo: interactive ? undefined : "skip",
      marker: { line: { color: PAPER, width: 0.6 } },
      colorbar: {
        title: { text: "Score PC1", font: { color: INK, size: 12 } },
        thickness: 12,
        len: 0.62,
        x: 0.99,
        xpad: 0,
        outlinewidth: 0,
        ticks: "outside",
        ticklen: 3,
        tickcolor: NODATA,
        tickfont: { color: INK, size: 10 }
      }
    }];
  }

  function buildLayout(interactive) {
    return {
      paper_bgcolor: PAPER,
      plot_bgcolor: PAPER,
      margin: { t: 6, r: 6, b: 6, l: 6 },
      font: { family: "Inter, ui-sans-serif, system-ui, sans-serif", color: INK },
      dragmode: interactive ? "pan" : false,
      geo: {
        scope: "world",
        projection: { type: "natural earth" },
        bgcolor: "rgba(0,0,0,0)",
        framewidth: 0,
        showframe: false,
        showcoastlines: false,
        showcountries: true,
        countrycolor: PAPER,
        countrywidth: 0.6,
        showland: true,
        landcolor: NODATA,
        showocean: false,
        showlakes: false,
        showrivers: false,
        lataxis: { showgrid: false },
        lonaxis: { showgrid: false, range: [-180, 180] }
      }
    };
  }

  function buildConfig(interactive) {
    if (!interactive) {
      return { staticPlot: true, displayModeBar: false, responsive: true };
    }
    return {
      responsive: true,
      scrollZoom: true,
      displaylogo: false,
      displayModeBar: true,
      modeBarButtonsToRemove: ["select2d", "lasso2d", "hoverClosestGeo", "toImage"]
    };
  }

  window.WorldMap = {
    render: function (el, opts) {
      opts = opts || {};
      var data = window.MAP_PC1_DATA || [];
      var interactive = !!opts.interactive;
      return Plotly.newPlot(
        el,
        buildTrace(data, interactive),
        buildLayout(interactive),
        buildConfig(interactive)
      );
    },
    purge: function (el) {
      if (window.Plotly && el) { Plotly.purge(el); }
    }
  };
}());
