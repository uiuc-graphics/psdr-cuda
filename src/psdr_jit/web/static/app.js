"use strict";
const form = document.querySelector("#controls");
const status = document.querySelector("#status");
const error = document.querySelector("#error");
const renderButton = document.querySelector("#render");
const labels = {x:"X", y:"Y", roughness:"roughness", albedo:"albedo", light:"light"};
let defaults, busy = false, snapshot = null;

function settings() {
  const result = Object.fromEntries(new FormData(form));
  for (const key in result) if (!["scene", "integrator", "derivative"].includes(key)) result[key] = Number(result[key]);
  return result;
}
function updateControls() {
  for (const input of form.querySelectorAll("input[type=range]")) {
    form.querySelector(`output[for=${input.id}]`).textContent = Number(input.value).toFixed(2);
    input.style.setProperty("--fill", `${100*(input.value-input.min)/(input.max-input.min)}%`);
  }
  if (!busy) {
    const dirty = snapshot && JSON.stringify(settings()) !== JSON.stringify(snapshot);
    document.body.classList.toggle("dirty", Boolean(dirty));
    status.textContent = dirty ? "Settings changed" : snapshot ? "Ready" : "Ready to render";
  }
}
function reset() {
  for (const [key,value] of Object.entries(defaults)) form.elements.namedItem(key).value = value;
  updateControls();
}
async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
async function render(event) {
  event?.preventDefault();
  if (busy || !defaults || !form.reportValidity()) return;
  busy = true;
  error.hidden = true;
  renderButton.disabled = true;
  document.body.classList.add("busy");
  document.body.classList.remove("dirty");
  const submitted = settings(), started = performance.now();
  const timer = setInterval(() => status.textContent = `Rendering · ${((performance.now()-started)/1000).toFixed(1)}s`, 100);
  try {
    const job = await request("/api/render", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(submitted)});
    let data;
    do {
      await new Promise(resolve => setTimeout(resolve, 400));
      data = await request(`/api/jobs/${job.id}`);
    } while (data.state === "running");
    if (data.state === "error") throw new Error(data.error);
    const result = data.result;
    for (const kind of ["render", "gradient"]) {
      const img = document.querySelector(`#${kind}-image`);
      img.src = result[kind]; img.hidden = false;
      const save = document.querySelector(`#save-${kind}`);
      save.href = result[kind]; save.hidden = false;
      save.download = `${submitted.scene}-${kind}-${submitted.derivative}.png`;
    }
    snapshot = submitted;
    document.querySelector("#gradient-title").textContent = `∂L / ∂${labels[submitted.derivative]}`;
    document.querySelector("#negative").textContent = `−${result.limit.toPrecision(3)}`;
    document.querySelector("#positive").textContent = `+${result.limit.toPrecision(3)}`;
    document.querySelector("#snapshot").textContent = `${submitted.scene === "cornell" ? "Cornell" : "Bunny"} · ${submitted.resolution}² · ${submitted.spp} spp · seed ${submitted.seed}`;
    document.querySelector("#timing").textContent = `${result.seconds.toFixed(2)}s`;
  } catch (failure) {
    error.textContent = failure.message; error.hidden = false;
  } finally {
    clearInterval(timer); busy = false; renderButton.disabled = false;
    document.body.classList.remove("busy"); updateControls();
    if (!error.hidden) status.textContent = "Render failed";
  }
}
form.addEventListener("input", updateControls);
form.addEventListener("submit", render);
document.querySelector("#reset").addEventListener("click", reset);
document.addEventListener("keydown", event => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") render(event); });
request("/api/config").then(config => { defaults = config.defaults; reset(); return render(); }).catch(failure => {
  error.textContent = failure.message; error.hidden = false; status.textContent = "Disconnected";
});
