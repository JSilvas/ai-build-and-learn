const grid = document.getElementById("grid");
const countEl = document.getElementById("count");

async function loadItems() {
  const res = await fetch("/api/items");
  const items = await res.json();
  render(items);
}

function render(items) {
  grid.innerHTML = "";
  countEl.textContent = `${items.length} generation${items.length === 1 ? "" : "s"}`;

  if (items.length === 0) {
    grid.innerHTML = '<p class="empty">No generations yet — run a cell in diffusers-testing.ipynb.</p>';
    return;
  }

  for (const item of items) {
    grid.appendChild(buildCard(item));
  }
}

function buildCard(item) {
  const card = document.createElement("div");
  card.className = "card";
  card.dataset.id = item.id;

  const media = item.type === "video"
    ? Object.assign(document.createElement("video"), {
        src: `/media/${item.filename}`,
        controls: true,
        loop: true,
        muted: true,
      })
    : Object.assign(document.createElement("img"), {
        src: `/media/${item.filename}`,
        alt: item.prompt,
        loading: "lazy",
      });

  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = item.type;

  const del = document.createElement("button");
  del.className = "delete";
  del.textContent = "×";
  del.title = "Remove";
  del.addEventListener("click", () => removeItem(item.id, card));

  const meta = document.createElement("div");
  meta.className = "meta";

  const prompt = document.createElement("p");
  prompt.className = "prompt";
  prompt.textContent = item.prompt;

  const model = document.createElement("p");
  model.className = "model";
  model.textContent = item.model;

  const params = document.createElement("div");
  params.className = "params";
  for (const [key, value] of Object.entries(item.params || {})) {
    const span = document.createElement("span");
    span.textContent = `${key}: ${value}`;
    params.appendChild(span);
  }

  meta.append(prompt, model, params);
  card.append(media, badge, del, meta);
  return card;
}

async function removeItem(id, card) {
  if (!confirm("Remove this generation?")) return;
  const res = await fetch(`/api/items/${id}`, { method: "DELETE" });
  if (res.ok) {
    card.remove();
    countEl.textContent = `${grid.children.length} generation${grid.children.length === 1 ? "" : "s"}`;
  }
}

loadItems();
