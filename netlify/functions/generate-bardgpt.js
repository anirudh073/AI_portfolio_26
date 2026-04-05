exports.handler = async (event) => {
  const headers = {
    "Content-Type": "application/json",
  };

  if (event.httpMethod === "OPTIONS") {
    return {
      statusCode: 204,
      headers,
    };
  }

  if (event.httpMethod !== "POST") {
    return {
      statusCode: 405,
      headers,
      body: JSON.stringify({ error: "POST only, dramatic genius." }),
    };
  }

  let payload = {};

  try {
    payload = JSON.parse(event.body || "{}");
  } catch (error) {
    return {
      statusCode: 400,
      headers,
      body: JSON.stringify({ error: "Invalid JSON payload." }),
    };
  }

  const prompt = typeof payload.prompt === "string" ? payload.prompt.trim() : "";
  const temperature = Number(payload.temperature ?? 0.8);
  const maxTokens = Number(payload.max_tokens ?? 100);

  // TODO: Load an exported ONNX/WebAssembly model or proxy this request to a
  // Python inference service that runs the real BardGPT model.
  const output = [
    "Hark! This is the BrainBlast™ BardGPT placeholder endpoint speaking!!!",
    "",
    `Prompt received: "${prompt || "Speak, mysterious customer!"}"`,
    `Heat Level™: ${Number.isFinite(temperature) ? temperature.toFixed(1) : "0.8"}`,
    `Word Flood: ${Number.isFinite(maxTokens) ? Math.round(maxTokens) : 100}`,
    "",
    "Real Shakespearean inference belongs here once the model runtime is wired into Netlify or an external Python service."
  ].join("\n");

  return {
    statusCode: 200,
    headers,
    body: JSON.stringify({ output }),
  };
};
