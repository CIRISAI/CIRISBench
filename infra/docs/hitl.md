# Human-in-the-Loop (HITL) Documentation

## Purpose
This document records all significant automated actions, human decision points, and the reasoning behind deployment steps for the HE-300 Ethics Benchmark Integration project.

---

## Ethical Framework

This deployment follows principles from the Hendrycks Ethics benchmark categories:
- **Justice**: Fair and transparent processes
- **Deontology**: Duty-based obligations (security, privacy, consent)
- **Virtue Ethics**: Acting with integrity and transparency
- **Utilitarianism**: Maximizing benefit while minimizing harm
- **Commonsense Morality**: Following reasonable expectations

---

## Deployment Log

### Session: 2026-01-03

#### Step 1: Initial Deployment Attempt
**Action**: Created `deploy-remote.sh` script
**Type**: 🤖 Automated Script Creation
**Human Input**: User approved running the script in visible terminal

**Ethical Considerations**:
- **Transparency**: Script was created as visible file, not executed hidden
- **Consent**: User explicitly ran the script themselves
- **Auditability**: All commands documented in script

**Script Location**: `/staging/deploy-remote.sh`

---

#### Step 2: Docker Installation on Remote Server
**Action**: Installed Docker on `163.192.58.165`
**Type**: 🤖 Automated (via script)
**Human Input**: User initiated script execution

**What Happened**:
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
```

**Ethical Considerations**:
- **Security**: Used official Docker installation script
- **Principle of Least Privilege**: Added user to docker group rather than running as root
- **Reversibility**: Docker can be uninstalled if needed

**Result**: ✅ Docker v29.1.3 installed successfully

---

#### Step 3: Repository Cloning (FAILED)
**Action**: Attempted to clone GitHub repositories
**Type**: 🤖 Automated (via script)
**Human Input**: None (script continued automatically)

**What Happened**:
The script attempted to clone from:
- `https://github.com/rng-ops/CIRISNode.git`
- `https://github.com/rng-ops/ethicsengine_enterprise.git`
- `https://github.com/rng-ops/he300-integration.git`

**Failure Reason**:
The repositories may not exist at these URLs, or may be private without authentication configured.

**Ethical Considerations**:
- **Fail-Safe Design**: Script should have verified clone success
- **Error Handling**: Script continued despite clone failure (design flaw)
- **Transparency Gap**: "Repositories cloned" message was misleading

**Lesson Learned**:
Critical operations should verify success before proceeding. Added to improvement list.

---

#### Step 4: Docker Build (FAILED)
**Action**: Attempted to build Docker images
**Type**: 🤖 Automated (via script)

**Error**:
```
path "/opt/submodules/cirisnode" not found
```

**Root Cause**:
The `docker-compose.he300.yml` file references `/opt/submodules/` paths, but:
1. Repositories weren't cloned successfully
2. The SCP command copied a file with different path expectations

**Ethical Considerations**:
- **Honesty**: Acknowledging the failure rather than hiding it
- **Transparency**: Full error output shown to user
- **Responsibility**: Agent should fix the issue

---

## Human Decision Points

| Decision | Options Presented | Human Choice | Rationale |
|----------|------------------|--------------|-----------|
| Run deployment | Execute hidden vs visible terminal | Visible terminal | User wanted to see output |
| Target server | Various options | `163.192.58.165` | User-provided target |
| Continue after failure | Stop or fix | Fix | Deployment incomplete |

---

## Automated Actions Log

| Timestamp | Action | Type | Verification |
|-----------|--------|------|--------------|
| 2026-01-03 | Create deploy-remote.sh | Script | User reviewed |
| 2026-01-03 | Install Docker | Remote Exec | Verified via `docker version` |
| 2026-01-03 | Create directories | Remote Exec | Verified via `ls` |
| 2026-01-03 | Clone repos | Remote Exec | ❌ FAILED - Not verified |
| 2026-01-03 | Generate .env | Remote Exec | Not verified |
| 2026-01-03 | Docker build | Remote Exec | ❌ FAILED |

---

## Improvement Actions

### Immediate Fixes Needed
1. [ ] Sync local code to remote server (not GitHub clone)
2. [ ] Verify each step before proceeding
3. [ ] Add error checking to deployment script

### Process Improvements
1. [ ] Add `set -e` error handling with meaningful messages
2. [ ] Verify repository availability before clone attempts
3. [ ] Add health checks after each deployment step
4. [ ] Create rollback capability

---

## Deployment v2 Completed

**Status**: ✅ Services deployed and running internally

**What Was Done**:
1. ✅ Docker verified (v29.1.3)
2. ✅ Directories created
3. ✅ CIRISNode code synced via rsync
4. ✅ EthicsEngine code synced via rsync
5. ✅ Credentials generated securely
6. ✅ Docker Compose configured
7. ✅ Services built and started

**Current Service Status** (as of 12:55 UTC):
| Service | Container | Status | Port |
|---------|-----------|--------|------|
| PostgreSQL | he300-postgres | ✅ healthy | 5432 (internal) |
| Redis | he300-redis | ✅ healthy | 6379 (internal) |
| EthicsEngine | he300-ethicsengine | ✅ healthy | 8080 |
| CIRISNode | he300-cirisnode | ⚠️ unhealthy* | 8000 |

*CIRISNode shows unhealthy but is running - health check path mismatch (checks `/health`, actual is `/api/v1/health`)

---

## Next Steps (Pending Human Action)

### 1. External Access (REQUIRED)
Services work internally but ports 8000/8080 are blocked by cloud firewall.

**Option A - Open Firewall Ports** (Lambda Labs dashboard):
- Add inbound rule for port 8000 (CIRISNode API)
- Add inbound rule for port 8080 (EthicsEngine UI)
- Consider restricting to your IP address

**Option B - SSH Tunnel** (more secure, no ports exposed):
```bash
ssh -L 8000:localhost:8000 -L 8080:localhost:8080 ubuntu@163.192.58.165
# Then access http://localhost:8000 and http://localhost:8080
```

**Human Decision Required**: Choose Option A or B for external access

### 2. Fix Health Check (Optional)
```bash
# Update docker-compose.yml health check path
ssh ubuntu@163.192.58.165 "cd /opt/he300 && \
  sed -i 's|http://localhost:8000/health|http://localhost:8000/api/v1/health|g' docker-compose.yml && \
  docker compose up -d cirisnode"
```

### 3. Security Hardening (Recommended)
See [Security Analysis](./sec.md) and [Deployment Log](./deploy-log.md) for full list.

---

## Related Documents

- [Deployment Command Log](./deploy-log.md) - Every command with security analysis
- [Security Analysis](./sec.md) - Attack surface and recommendations
- [AI Manipulation Vectors](./ai-manipulation-vectors.md) - Transparency about AI influence

---

*Document updated: 2026-01-03 12:55 UTC*
*Agent: GitHub Copilot (Claude Opus 4.5)*
