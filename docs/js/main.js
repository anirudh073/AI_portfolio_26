document.addEventListener("DOMContentLoaded", () => {
  initCatalogueFilters();
  initChatDemos();
  initShowroomGalleries();
});

function initCatalogueFilters() {
  const filterButtons = Array.from(document.querySelectorAll("[data-filter]"));
  const cards = Array.from(document.querySelectorAll("[data-category]"));
  const status = document.querySelector("[data-filter-status]");

  if (!filterButtons.length || !cards.length) {
    return;
  }

  const updateFilter = (filter) => {
    let visibleCount = 0;

    cards.forEach((card) => {
      const matches = filter === "all" || card.dataset.category === filter;
      card.classList.toggle("is-hidden", !matches);

      if (matches) {
        visibleCount += 1;
      }
    });

    filterButtons.forEach((button) => {
      const active = button.dataset.filter === filter;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });

    if (status) {
      const label = filter === "all" ? "ALL DEPARTMENTS" : `${filter.toUpperCase()} DEPARTMENT`;
      status.textContent = `${visibleCount} DEAL${visibleCount === 1 ? "" : "S"} IN ${label}!`;
    }
  };

  filterButtons.forEach((button) => {
    button.addEventListener("click", () => {
      updateFilter(button.dataset.filter || "all");
    });
  });

  updateFilter("all");
}

function initChatDemos() {
  const chatDemos = Array.from(document.querySelectorAll("[data-chat-demo]"));

  chatDemos.forEach((demo) => {
    const log = demo.querySelector("[data-chat-log]");
    const form = demo.querySelector("[data-chat-form]");
    const promptInput = demo.querySelector("[data-chat-prompt]");
    const temperatureInput = demo.querySelector("[data-chat-temperature]");
    const temperatureValue = demo.querySelector("[data-chat-temperature-value]");
    const maxTokensInput = demo.querySelector("[data-chat-max-tokens]");
    const sendButton = demo.querySelector("[data-chat-submit]");
    const endpoint = demo.dataset.endpoint;
    const welcome = demo.dataset.welcome;
    const productName = demo.dataset.productName || "BrainBlast Bot";

    if (!log || !form || !promptInput || !temperatureInput || !maxTokensInput || !endpoint) {
      return;
    }

    const appendMessage = (role, text) => {
      const message = document.createElement("article");
      const tag = document.createElement("span");
      const body = document.createElement("p");

      message.className = `chat-message chat-message--${role}`;
      tag.className = "chat-message-tag";
      body.className = "chat-message-body";

      if (role === "user") {
        tag.textContent = "YOU";
      } else if (role === "system") {
        tag.textContent = "SYSTEM";
      } else {
        tag.textContent = productName.toUpperCase();
      }

      body.textContent = text;
      message.append(tag, body);
      log.appendChild(message);
      log.scrollTop = log.scrollHeight;
    };

    const syncTemperature = () => {
      if (temperatureValue) {
        temperatureValue.textContent = Number(temperatureInput.value).toFixed(1);
      }
    };

    syncTemperature();
    temperatureInput.addEventListener("input", syncTemperature);

    if (welcome) {
      appendMessage("bot", welcome);
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      const prompt = promptInput.value.trim();

      if (!prompt) {
        return;
      }

      appendMessage("user", prompt);
      promptInput.value = "";
      promptInput.focus();

      if (sendButton) {
        sendButton.disabled = true;
        sendButton.textContent = "GENERATING!!!";
      }

      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            prompt,
            temperature: Number(temperatureInput.value),
            max_tokens: Number(maxTokensInput.value),
          }),
        });

        if (!response.ok) {
          throw new Error(`Function returned ${response.status}`);
        }

        const payload = await response.json();
        appendMessage("bot", payload.output || "The BrainBlast™ signal came back blank. Try again with more drama!!!");
      } catch (error) {
        appendMessage(
          "system",
          "Transmission jammed! The serverless gremlins dropped the cable. Wire up the real inference endpoint and try again!!!"
        );
      } finally {
        if (sendButton) {
          sendButton.disabled = false;
          sendButton.textContent = "GENERATE!!!";
        }
      }
    });
  });
}

function initShowroomGalleries() {
  const galleries = Array.from(document.querySelectorAll("[data-gallery]"));

  galleries.forEach((gallery) => {
    const stage = gallery.querySelector("[data-gallery-stage]");
    const caption = gallery.querySelector("[data-gallery-caption]");
    const title = gallery.querySelector("[data-gallery-title]");
    const items = Array.from(gallery.querySelectorAll("[data-gallery-item]"));

    if (!stage || !caption || !title || !items.length) {
      return;
    }

    const renderItem = (item) => {
      items.forEach((candidate) => {
        const active = candidate === item;
        candidate.classList.toggle("is-active", active);
        candidate.setAttribute("aria-pressed", String(active));
      });

      stage.innerHTML = "";
      title.textContent = item.dataset.galleryTitle || "Featured Model";
      caption.textContent = item.dataset.galleryCaption || "";

      if (item.dataset.galleryType === "video") {
        const video = document.createElement("video");
        video.controls = true;
        video.muted = true;
        video.playsInline = true;
        video.preload = "metadata";
        video.className = "showroom-stage-video";

        const source = document.createElement("source");
        source.src = item.dataset.gallerySrc || "";
        source.type = "video/mp4";

        video.appendChild(source);
        stage.appendChild(video);
        return;
      }

      const placeholder = document.createElement("div");
      const label = document.createElement("span");

      placeholder.className = "media-placeholder hazard showroom-stage-placeholder";
      label.textContent = item.dataset.galleryPlaceholder || "COMING SOON — [CLASSIFIED]";

      placeholder.appendChild(label);
      stage.appendChild(placeholder);
    };

    items.forEach((item) => {
      item.addEventListener("click", () => {
        renderItem(item);
      });
    });

    renderItem(items[0]);
  });
}
