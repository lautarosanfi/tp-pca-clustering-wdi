(function () {
  const registry = {};

  function ensureLightbox() {
    let node = document.querySelector(".lightbox");
    if (node) return node;

    node = document.createElement("div");
    node.className = "lightbox";
    node.innerHTML = '<button class="lightbox__close" type="button" aria-label="Cerrar">Cerrar</button><img alt="">';
    document.body.appendChild(node);

    const close = () => node.classList.remove("is-open");
    node.querySelector(".lightbox__close").addEventListener("click", close);
    node.addEventListener("click", (event) => {
      if (event.target === node) close();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") close();
    });

    return node;
  }

  function openImage(src, alt) {
    const lightbox = ensureLightbox();
    const img = lightbox.querySelector("img");
    img.src = src;
    img.alt = alt || "";
    lightbox.classList.add("is-open");
  }

  function enhanceFigures() {
    document.querySelectorAll(".figure-card").forEach((figure) => {
      const img = figure.querySelector("img");
      if (!img) return;

      const actions = document.createElement("div");
      actions.className = "figure-actions";

      const expand = document.createElement("button");
      expand.type = "button";
      expand.textContent = "Ampliar";
      expand.addEventListener("click", () => openImage(img.currentSrc || img.src, img.alt));

      actions.appendChild(expand);
      figure.appendChild(actions);
    });
  }

  window.ReportFigures = {
    register(id, render) {
      registry[id] = render;
    },
    render(id, target, data) {
      if (registry[id]) {
        registry[id](target, data);
      }
    },
    registry
  };

  enhanceFigures();
}());
