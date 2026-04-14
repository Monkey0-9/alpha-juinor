import { createServer } from "node:http";
import { appendFile, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { WebSocket, WebSocketServer } from "ws";

type Tick = {
  symbol: string;
  price: number;
  bid?: number;
  ask?: number;
  volume?: number;
  ts: number;
  source?: string;
};

const port = Number(process.env.TS_GATEWAY_PORT ?? "8787");
const persist = process.env.TS_GATEWAY_PERSIST === "1";
const runtimePath = process.env.TS_GATEWAY_JSONL_PATH ?? join("runtime", "live_ticks.jsonl");

const latestBySymbol = new Map<string, Tick>();
const streamClients = new Set<WebSocket>();

const server = createServer((req, res) => {
  if (req.url === "/health") {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(
      JSON.stringify({
        ok: true,
        stream_clients: streamClients.size,
        tracked_symbols: latestBySymbol.size
      })
    );
    return;
  }

  res.writeHead(404);
  res.end("not found");
});

const ingestWss = new WebSocketServer({ noServer: true });
const streamWss = new WebSocketServer({ noServer: true });

function isTick(payload: unknown): payload is Tick {
  if (!payload || typeof payload !== "object") return false;
  const obj = payload as Record<string, unknown>;
  return (
    typeof obj.symbol === "string" &&
    typeof obj.price === "number" &&
    typeof obj.ts === "number"
  );
}

function encodeTick(tick: Tick): string {
  return JSON.stringify({
    t: "tick",
    s: tick.symbol,
    p: tick.price,
    b: tick.bid ?? null,
    a: tick.ask ?? null,
    v: tick.volume ?? null,
    ts: tick.ts,
    src: tick.source ?? "unknown"
  });
}

async function persistTick(tick: Tick): Promise<void> {
  if (!persist) return;
  await mkdir(dirname(runtimePath), { recursive: true });
  await appendFile(runtimePath, `${JSON.stringify(tick)}\n`, "utf8");
}

function broadcastTick(tick: Tick): void {
  const msg = encodeTick(tick);
  for (const client of streamClients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(msg);
    }
  }
}

ingestWss.on("connection", (socket) => {
  socket.on("message", async (data) => {
    try {
      const raw = typeof data === "string" ? data : data.toString("utf8");
      const parsed = JSON.parse(raw) as unknown;
      if (!isTick(parsed)) return;

      latestBySymbol.set(parsed.symbol, parsed);
      broadcastTick(parsed);
      await persistTick(parsed);
    } catch {
      // Ignore malformed messages to avoid backpressure from bad producers.
    }
  });
});

streamWss.on("connection", (socket) => {
  streamClients.add(socket);

  const snapshot = Array.from(latestBySymbol.values()).map((tick) => ({
    s: tick.symbol,
    p: tick.price,
    b: tick.bid ?? null,
    a: tick.ask ?? null,
    v: tick.volume ?? null,
    ts: tick.ts,
    src: tick.source ?? "unknown"
  }));
  socket.send(JSON.stringify({ t: "snapshot", ticks: snapshot }));

  socket.on("close", () => {
    streamClients.delete(socket);
  });
});

server.on("upgrade", (request, socket, head) => {
  const path = request.url ?? "";
  if (path.startsWith("/ingest")) {
    ingestWss.handleUpgrade(request, socket, head, (ws) => {
      ingestWss.emit("connection", ws, request);
    });
    return;
  }
  if (path.startsWith("/stream")) {
    streamWss.handleUpgrade(request, socket, head, (ws) => {
      streamWss.emit("connection", ws, request);
    });
    return;
  }
  socket.destroy();
});

server.listen(port, "0.0.0.0", () => {
  // eslint-disable-next-line no-console
  console.log(`TS live-data gateway listening on :${port}`);
  // eslint-disable-next-line no-console
  console.log(`Ingest endpoint: ws://127.0.0.1:${port}/ingest`);
  // eslint-disable-next-line no-console
  console.log(`Stream endpoint: ws://127.0.0.1:${port}/stream`);
});
