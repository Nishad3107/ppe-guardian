# PPE Guardian Improvement Roadmap

## Phase 1: Correctness and Configuration
- Replace `models/ppe.pt` with a real PPE-trained model that exposes `helmet`, `mask`, `glasses`, and `boots`.
- Keep `REQUIRE_VALID_PPE_MODEL=true` in `.env.local` when validating deployment environments.
- Add a startup checklist that verifies the person model path, PPE model path, and recorded video path.

## Phase 2: Architecture
- Move frame processing into a dedicated worker process instead of running inference inside Flask request handlers.
- Remove global shared source state so one user switching inputs does not affect every other user.
- Introduce a source/job abstraction for camera, recorded video, and uploaded video sessions.

## Phase 3: Persistence and Observability
- Replace `violations.log` with SQLite or Postgres for structured querying and retention.
- Store upload metadata and retention timestamps in the database.
- Add a health endpoint that reports model status, source status, and last processed frame time.

## Phase 4: Security and Operations
- Add upload size limits, MIME validation, and cleanup for stale uploaded videos.
- Protect dashboard and upload routes with authentication.
- Run the app behind a production WSGI server instead of Flask's built-in development server.

## Phase 5: Accuracy and Performance
- Run PPE detection on person or upper-body crops instead of the full frame.
- Add benchmark scripts for FPS, average inference time, and violation precision/recall.
- Introduce sample evaluation videos with expected outputs for regression testing.

## Phase 6: Testing and Release Discipline
- Add tests for startup validation, source fallback, upload handling, metrics, and violation parsing.
- Pin Python and Node dependencies with lockfiles.
- Add CI checks for syntax, tests, and model configuration validation.
