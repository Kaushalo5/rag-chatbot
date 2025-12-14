// frontend/app.js
// Simple UI client for the Generate -> Approve -> Query flow.
// Assumes backend at http://localhost:8000

const backend = "http://localhost:8000";

function by(id){ return document.getElementById(id); }

async function postJson(path, body){
  const resp = await fetch(backend + path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  if(!resp.ok){
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }
  return resp.json();
}

/* ---------- Generate ---------- */
by("generateBtn").addEventListener("click", async () => {
  const prompt = by("generatePrompt").value.trim();
  if(!prompt) return alert("Write a prompt first.");
  by("genStatus").textContent = "Generating…";
  try{
    const j = await postJson("/api/generate-doc", {prompt});
    by("draftPreview").value = j.draft || "";
    by("genStatus").textContent = "Draft ready — review below.";
  }catch(err){
    by("genStatus").textContent = "Generation failed: " + err.message;
  }
});

/* ---------- Approve & Add to KB ---------- */
by("approveBtn").addEventListener("click", async () => {
  const doc_id = by("docIdForApprove").value.trim();
  const text = by("draftPreview").value.trim();
  if(!doc_id) return alert("Enter doc_id before approving.");
  if(!text) return alert("Draft is empty.");
  by("approveStatus").textContent = "Indexing…";
  try{
    const j = await postJson("/api/approve-doc", {doc_id, text});
    by("approveStatus").textContent = `Indexed (${j.chunks_added} chunks).`;
  }catch(err){
    by("approveStatus").textContent = "Index failed: " + err.message;
  }
});

/* ---------- Manual index one-liner ---------- */
by("manualIndexBtn").addEventListener("click", async () => {
  const doc_id = by("manualDocId").value.trim();
  const text = by("manualText").value.trim();
  if(!doc_id || !text) return alert("doc_id and text required.");
  by("manualStatus").textContent = "Indexing…";
  try{
    const j = await postJson("/api/embed-docs", {doc_id, text});
    by("manualStatus").textContent = `Indexed (${j.chunks_added} chunks).`;
  }catch(err){
    by("manualStatus").textContent = "Index failed: " + err.message;
  }
});

/* ---------- Query ---------- */
by("askBtn").addEventListener("click", async () => {
  const q = by("queryInput").value.trim();
  if(!q) return alert("Write a question.");
  by("askStatus").textContent = "Thinking…";
  by("answerBox").innerHTML = "";
  try{
    const j = await postJson("/api/query", {query: q, top_k: 3});
    // Display answer, contexts, and meta
    const ansHtml = `<div class="answer"><strong>Answer</strong><pre>${escapeHtml(j.answer)}</pre></div>`;
    let ctxHtml = "<div class='contexts'><strong>Retrieved contexts</strong>";
    (j.contexts||[]).slice(0,5).forEach((c,i)=>{
      ctxHtml += `<div class='ctx'><em>[${i+1}]</em> ${escapeHtml((c.text||"").slice(0,300))} <span class='score'>(${(c.score||0).toFixed(3)})</span></div>`;
    });
    ctxHtml += "</div>";
    const metaHtml = `<div class='meta'><strong>Meta</strong><pre>${escapeHtml(JSON.stringify(j.meta||{}, null, 2))}</pre></div>`;
    by("answerBox").innerHTML = ansHtml + ctxHtml + metaHtml;
    by("askStatus").textContent = "Done";
  }catch(err){
    by("askStatus").textContent = "Query failed: " + err.message;
  }
});

/* ---------- Helpers ---------- */
function escapeHtml(s){
  if(!s) return "";
  return s.replace(/[&<>"']/g, (m)=>({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" })[m]);
}
