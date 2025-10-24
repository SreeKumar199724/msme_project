# How to Rebuild and Update msme_chatbot_agents Package

## Quick Rebuild Process

After making changes to any of these files:
- `mcp_agents.py`
- `mcp_agent_tools.py`
- `__init__.py`
- `pyproject.toml`

Follow these steps to rebuild and reinstall:

### Step 1: Navigate to Package Directory
```bash
cd C:\Users\user\PROJECT-004-MSME-CHATBOT-QA\msme_chatbot_qa\msme_chatbot_agents
```

### Step 2: Clean Old Build (Optional but Recommended)

**PowerShell:**
```powershell
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
```

**Git Bash/Unix-like shells:**
```bash
rm -rf dist
```

**CMD:**
```cmd
rmdir /S /Q dist
```

### Step 3: Build New Wheel
```bash
uv build
```

This creates:
- `dist/msme_chatbot_agents-0.1.0-py3-none-any.whl` (wheel file)
- `dist/msme_chatbot_agents-0.1.0.tar.gz` (source distribution)

### Step 4: Reinstall in Main VEnv
```bash
cd ..
uv pip install --force-reinstall msme_chatbot_agents/dist/msme_chatbot_agents-0.1.0-py3-none-any.whl
```

Or in one command from the main project directory:
```bash
cd C:\Users\user\PROJECT-004-MSME-CHATBOT-QA\msme_chatbot_qa
uv pip install --force-reinstall msme_chatbot_agents/dist/msme_chatbot_agents-0.1.0-py3-none-any.whl
```

### Step 5: Verify Installation
```bash
.venv\Scripts\python.exe -c "import msme_chatbot_agents; print('Package updated successfully!')"
```

## Complete One-Liner Commands

### From Package Directory

**PowerShell:**
```powershell
cd C:\Users\user\PROJECT-004-MSME-CHATBOT-QA\msme_chatbot_qa\msme_chatbot_agents; Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue; uv build; cd ..; uv pip install --force-reinstall msme_chatbot_agents/dist/msme_chatbot_agents-0.1.0-py3-none-any.whl
```

**Git Bash:**
```bash
cd C:\Users\user\PROJECT-004-MSME-CHATBOT-QA\msme_chatbot_qa\msme_chatbot_agents && rm -rf dist && uv build && cd .. && uv pip install --force-reinstall msme_chatbot_agents/dist/msme_chatbot_agents-0.1.0-py3-none-any.whl
```

### From Main Project Directory

**PowerShell:**
```powershell
cd C:\Users\user\PROJECT-004-MSME-CHATBOT-QA\msme_chatbot_qa; Remove-Item -Recurse -Force msme_chatbot_agents/dist -ErrorAction SilentlyContinue; cd msme_chatbot_agents; uv build; cd ..; uv pip install --force-reinstall msme_chatbot_agents/dist/msme_chatbot_agents-0.1.0-py3-none-any.whl
```

**Git Bash:**
```bash
cd C:\Users\user\PROJECT-004-MSME-CHATBOT-QA\msme_chatbot_qa && rm -rf msme_chatbot_agents/dist && cd msme_chatbot_agents && uv build && cd .. && uv pip install --force-reinstall msme_chatbot_agents/dist/msme_chatbot_agents-0.1.0-py3-none-any.whl
```

## What Each Command Does

- `Remove-Item -Recurse -Force dist` (PowerShell) or `rm -rf dist` (Bash) - Removes old build artifacts
- `uv build` - Builds both wheel (.whl) and source (.tar.gz) distribution
- `uv pip install --force-reinstall <wheel>` - Reinstalls the wheel, replacing the old version

## When to Rebuild

Rebuild whenever you make changes to:
1. **Code files**: `mcp_agents.py`, `mcp_agent_tools.py`, `__init__.py`
2. **Configuration**: `pyproject.toml` (especially dependencies)
3. **Version**: When you bump the version number

## Alternative: Editable Install (For Development)

If you're actively developing and want changes to reflect immediately without rebuilding:

```bash
cd C:\Users\user\PROJECT-004-MSME-CHATBOT-QA\msme_chatbot_qa
uv pip install -e msme_chatbot_agents
```

**Note**: Editable install had issues in this project, so wheel installation is recommended.

## Troubleshooting

### Build Fails
```bash
# Check pyproject.toml syntax
cd msme_chatbot_agents
cat pyproject.toml
```

### Import Fails After Install
```bash
# Check if files are in site-packages
ls -la .venv/Lib/site-packages/msme_chatbot_agents/

# Should show:
# __init__.py
# mcp_agents.py
# mcp_agent_tools.py
```

### Wrong Version Installed
```bash
# Check installed version
uv pip list | grep msme

# Uninstall and reinstall
uv pip uninstall msme-chatbot-agents
uv pip install msme_chatbot_agents/dist/msme_chatbot_agents-0.1.0-py3-none-any.whl
```

## Version Bumping

To update the version (e.g., from 0.1.0 to 0.1.1):

1. Edit `pyproject.toml`:
   ```toml
   [project]
   name = "msme-chatbot-agents"
   version = "0.1.1"  # Change this
   ```

2. Edit `__init__.py`:
   ```python
   __version__ = '1.0.1'  # Change this
   ```

3. Rebuild and reinstall:
   ```bash
   cd msme_chatbot_agents && rm -rf dist && uv build && cd .. && uv pip install --force-reinstall msme_chatbot_agents/dist/msme_chatbot_agents-0.1.1-py3-none-any.whl
   ```
