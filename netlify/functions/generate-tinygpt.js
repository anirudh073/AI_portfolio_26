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
      body: JSON.stringify({ error: "POST only, tiny titan." }),
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
  // Python inference service that runs the real TinyGPT model.
  const output = [
    "TinyGPT placeholder response incoming at an unreasonable volume!!!",
    "",
    `Prompt received: "${prompt || "Give me something delightfully compact!"}"`,
    `Heat Level™: ${Number.isFinite(temperature) ? temperature.toFixed(1) : "0.8"}`,
    `Word Flood: ${Number.isFinite(maxTokens) ? Math.round(maxTokens) : 100}`,
    "",
    "Replace this stub with real TinyGPT inference when the model runtime is ready."
  ].join("\n");

  return {
    statusCode: 200,
    headers,
    body: JSON.stringify({ output }),
  };
};
