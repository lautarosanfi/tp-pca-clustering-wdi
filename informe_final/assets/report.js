/*
 * report.js — Interactividad del informe.
 *
 * Modelo: las figuras se muestran ESTÁTICAS en la página (imagen o mapa en modo
 * staticPlot). La interactividad total (hover, zoom, paneo) se habilita SOLO al
 * ampliar una figura en un modal; al cerrarlo, la figura vuelve a su estado estático.
 *
 * Además: carga diferida de Plotly (solo necesaria para el mapa) y apertura
 * automática de <details> al navegar por anclas.
 */
(function () {
  "use strict";

  var EXPAND_ICON =
    '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3m13-5v3a2 2 0 0 1-2 2h-3"/>' +
    '</svg>';

  /* ─────────────────────────── Carga diferida de Plotly ─────────────────────────── */
  // Plotly solo se necesita para el mapa: se inyecta una única vez, bajo demanda.
  var plotlyPromise = null;
  function loadPlotly() {
    if (window.Plotly) return Promise.resolve(window.Plotly);
    if (plotlyPromise) return plotlyPromise;
    plotlyPromise = new Promise(function (resolve, reject) {
      var s = document.createElement("script");
      s.src = "figures/plotly.min.js";
      s.async = true;
      s.onload = function () { resolve(window.Plotly); };
      s.onerror = function () { reject(new Error("No se pudo cargar Plotly")); };
      document.head.appendChild(s);
    });
    return plotlyPromise;
  }

  /* ───────────────────────────────── Modal único ────────────────────────────────── */
  var modal, modalStage, modalTitle, lastFocus;

  function buildModal() {
    modal = document.createElement("div");
    modal.className = "modal";
    modal.setAttribute("aria-hidden", "true");
    modal.innerHTML =
      '<div class="modal__backdrop" data-close></div>' +
      '<div class="modal__dialog" role="dialog" aria-modal="true" aria-label="Figura ampliada">' +
        '<header class="modal__bar">' +
          '<p class="modal__title"></p>' +
          '<button class="modal__close" type="button" data-close aria-label="Cerrar (Esc)">' +
            'Cerrar <span aria-hidden="true">✕</span></button>' +
        '</header>' +
        '<div class="modal__stage"></div>' +
      '</div>';
    document.body.appendChild(modal);
    modalStage = modal.querySelector(".modal__stage");
    modalTitle = modal.querySelector(".modal__title");

    modal.addEventListener("click", function (e) {
      if (e.target.hasAttribute("data-close")) closeModal();
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && modal.classList.contains("is-open")) closeModal();
    });
  }

  function openModal(title) {
    if (!modal) buildModal();
    lastFocus = document.activeElement;
    modalTitle.textContent = title || "";
    modalStage.innerHTML = "";
    modal.classList.add("is-open");
    modal.setAttribute("aria-hidden", "false");
    document.body.classList.add("is-modal-open");
    modal.querySelector(".modal__close").focus();
    return modalStage;
  }

  function closeModal() {
    if (!modal) return;
    // Vaciar el escenario detiene los iframes (descarga su Plotly) y libera el mapa.
    var mapEl = modalStage.querySelector(".map-mount");
    if (mapEl && window.WorldMap) WorldMap.purge(mapEl);
    modalStage.innerHTML = "";
    modal.classList.remove("is-open");
    modal.setAttribute("aria-hidden", "true");
    document.body.classList.remove("is-modal-open");
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  /* ──────────────────────────── Apertura por tipo de figura ─────────────────────── */
  function figureTitle(card) {
    var strong = card.querySelector("figcaption strong");
    var label = strong ? strong.textContent.trim() : "";
    var img = card.querySelector("img.figure-img:not(.map-print)");
    var alt = img ? img.getAttribute("alt") : (card.getAttribute("data-title") || "");
    return (label + " " + alt).replace(/\s+/g, " ").trim();
  }

  function openIframeChart(card) {
    var src = card.getAttribute("data-interactive");
    var h = parseInt(card.getAttribute("data-fig-height"), 10) || 640;
    var stage = openModal(figureTitle(card));
    var frame = document.createElement("iframe");
    frame.className = "modal__frame";
    frame.setAttribute("title", figureTitle(card) || "Gráfico interactivo");
    frame.style.height = h + "px";
    frame.src = src;
    stage.appendChild(frame);
  }

  function openMapChart(card) {
    var stage = openModal(figureTitle(card));
    var mount = document.createElement("div");
    mount.className = "map-mount map-mount--modal";
    stage.appendChild(mount);
    loadPlotly()
      .then(function () { return WorldMap.render(mount, { interactive: true }); })
      .catch(function () {
        mount.innerHTML = '<p class="map-error">No se pudo cargar el mapa interactivo.</p>';
      });
  }

  /* ─────────────────────────── Mejora de figuras (botón Ampliar) ─────────────────── */
  function enhanceFigures() {
    document.querySelectorAll(".figure-card").forEach(function (card) {
      var isMap = card.getAttribute("data-chart") === "map-pc1";
      var isChart = card.hasAttribute("data-interactive");
      if (!isMap && !isChart) return;

      var open = function () { isMap ? openMapChart(card) : openIframeChart(card); };

      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "figure-expand";
      btn.innerHTML = EXPAND_ICON + "<span>Ampliar</span>";
      btn.setAttribute("aria-label", "Ampliar figura");
      btn.addEventListener("click", function (e) { e.stopPropagation(); open(); });

      var media = card.querySelector(".figure-media") || card;
      media.appendChild(btn);
      card.classList.add("is-zoomable");

      // El área visual también abre la ampliación (con soporte de teclado).
      var target = isMap ? card.querySelector(".map-mount") : card.querySelector("img.figure-img");
      if (target) {
        target.addEventListener("click", open);
        target.setAttribute("tabindex", "0");
        target.setAttribute("role", "button");
        target.setAttribute("aria-label", "Ampliar figura");
        target.addEventListener("keydown", function (e) {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); }
        });
      }
    });
  }

  /* ───────────────────────── Mapa estático en la página (diferido) ───────────────── */
  function initInlineMap() {
    var card = document.querySelector('[data-chart="map-pc1"]');
    if (!card) return;
    var mount = card.querySelector(".map-mount");
    if (!mount) return;

    var done = false;
    var render = function () {
      if (done) return;
      done = true;
      loadPlotly()
        .then(function () { return WorldMap.render(mount, { interactive: false }); })
        .then(function () { card.classList.add("map-ready"); })
        .catch(function () { card.classList.add("map-ready"); });
    };

    // Render diferido tras la primera pintura: no bloquea la carga inicial y
    // no depende del scroll (el observador de intersección puede no dispararse).
    if ("requestIdleCallback" in window) {
      requestIdleCallback(render, { timeout: 1500 });
    } else {
      setTimeout(render, 250);
    }
  }

  /* ──────────── Abrir <details> al navegar a un ancla interna (referencias) ──────── */
  function openTargetDetails() {
    var hash = window.location.hash;
    if (!hash || hash.length < 2) return;
    var id = hash.slice(1);
    var target;
    try { target = document.getElementById(decodeURIComponent(id)); }
    catch (e) { target = document.getElementById(id); }
    if (!target) return;

    var node = target;
    while (node) {
      if (node.tagName === "DETAILS" && !node.open) node.open = true;
      node = node.parentElement;
    }
    requestAnimationFrame(function () {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  /* ─────────────────────────────────── Arranque ─────────────────────────────────── */
  enhanceFigures();
  initInlineMap();
  openTargetDetails();
  window.addEventListener("hashchange", openTargetDetails);
}());
