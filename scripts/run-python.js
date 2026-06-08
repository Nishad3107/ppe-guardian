const { existsSync } = require("node:fs");
const path = require("node:path");
const { spawn } = require("node:child_process");

const repoRoot = path.resolve(__dirname, "..");
const mplConfigDir = path.join(repoRoot, ".cache", "matplotlib");

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
if (!pythonCmd) {
  console.error("No Python interpreter found. Create a local venv or install python3.");
  process.exit(1);
}

const child = spawn(pythonCmd, process.argv.slice(2), {
  cwd: repoRoot,
  stdio: "inherit",
  env: {
    ...process.env,
    PYTHONUNBUFFERED: "1",
    MPLCONFIGDIR: process.env.MPLCONFIGDIR || mplConfigDir,
  },
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});

child.on("error", (error) => {
  console.error(`Failed to start Python with '${pythonCmd}':`, error.message);
  process.exit(1);
});
