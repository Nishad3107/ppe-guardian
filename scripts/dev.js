const { existsSync } = require("node:fs");
const path = require("node:path");
const net = require("node:net");
const { spawn } = require("node:child_process");

const repoRoot = path.resolve(__dirname, "..");

function detectPython() {
  const candidates = [
    path.join(repoRoot, "venv", "bin", "python"),
    path.join(repoRoot, "venv", "Scripts", "python.exe"),
    path.join(repoRoot, "source", "bin", "python"),
    path.join(repoRoot, "source", "Scripts", "python.exe"),
    "python3",
    "python",
  ];

  return candidates.find((candidate) => {
    if (candidate.includes(path.sep)) {
      return existsSync(candidate);
    }
    return true;
  });
}

const pythonCmd = detectPython();
const host = process.env.SERVER_HOST || "127.0.0.1";
const requestedPort = Number.parseInt(process.env.SERVER_PORT || "8000", 10);

if (!pythonCmd) {
  console.error("No Python interpreter found. Create a local venv or install python3.");
  process.exit(1);
}

function isPortFree(hostname, port) {
  return new Promise((resolve) => {
    const server = net.createServer();

    server.once("error", () => resolve(false));
    server.once("listening", () => {
      server.close(() => resolve(true));
    });
    server.listen(port, hostname);
  });
}

async function findAvailablePort(hostname, startPort) {
  if (Number.isNaN(startPort) || startPort <= 0) {
    return 8000;
  }

  for (let port = startPort; port < startPort + 25; port += 1) {
    // eslint-disable-next-line no-await-in-loop
    if (await isPortFree(hostname, port)) {
      return port;
    }
  }

  throw new Error(`No free port found between ${startPort} and ${startPort + 24}.`);
}

async function main() {
  const port = process.env.SERVER_PORT
    ? requestedPort
    : await findAvailablePort(host, requestedPort);

  if (!process.env.SERVER_PORT && port !== requestedPort) {
    console.log(`Port ${requestedPort} is busy. Starting PPE Guardian on http://${host}:${port}`);
  }

  const child = spawn(pythonCmd, ["app.py"], {
    cwd: repoRoot,
    stdio: "inherit",
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
      SERVER_HOST: host,
      SERVER_PORT: String(port),
    },
  });

  const forwardSignal = (signal) => {
    if (!child.killed) {
      child.kill(signal);
    }
  };

  process.on("SIGINT", () => forwardSignal("SIGINT"));
  process.on("SIGTERM", () => forwardSignal("SIGTERM"));

  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 0);
  });

  child.on("error", (error) => {
    console.error(`Failed to start Python app with '${pythonCmd}':`, error.message);
    process.exit(1);
  });
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
