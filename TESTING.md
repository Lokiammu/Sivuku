# Testing Guide for deploy_to_hf.sh

## Local Script Testing

### 1. Dry-Run Test (Safest - No Deployment)

Test the script without actually deploying:

```bash
# Test with dry-run (prepares files but doesn't upload)
./scripts/deploy_to_hf.sh echo_env --dry-run

# Verify the staging directory was created
ls -la hf-staging/

# Check the generated files
tree hf-staging/ -L 3
```

**What to verify:**
- ✅ Staging directory created at `hf-staging/<namespace>/<env_name>`
- ✅ `Dockerfile` exists and looks correct
- ✅ `README.md` exists with correct metadata
- ✅ `src/core/` and `src/envs/<env>/` are copied correctly
- ✅ No deployment attempted (no HF API calls)

### 2. Test with Test Namespace

Deploy to your personal test space:

```bash
# Use your HF username and add -test suffix to avoid affecting production
./scripts/deploy_to_hf.sh echo_env \
  --hf-namespace $(hf auth whoami | head -n1) \
  --space-suffix -test

# Or use positional arguments
./scripts/deploy_to_hf.sh echo_env "" "$(hf auth whoami | head -n1)"
```

**What to verify:**
- ✅ Space created at `https://huggingface.co/spaces/<your-username>/echo_env-test`
- ✅ Files uploaded correctly
- ✅ Space builds successfully (check Space logs)

### 3. Test Different Environments

```bash
# Test smallest environment first (echo_env)
./scripts/deploy_to_hf.sh echo_env --dry-run

# Test with specific base image SHA
./scripts/deploy_to_hf.sh echo_env --base-sha abc123 --dry-run

# Test other environments
./scripts/deploy_to_hf.sh coding_env --dry-run
./scripts/deploy_to_hf.sh chat_env --dry-run
```

### 4. Validate Generated Files

After dry-run, inspect the generated files:

```bash
# Check Dockerfile
cat hf-staging/<namespace>/echo_env/Dockerfile

# Check README
cat hf-staging/<namespace>/echo_env/README.md

# Verify structure
find hf-staging/<namespace>/echo_env -type f
```

## GitHub Actions Testing

### 0. Do I Need to Merge to Main?

**No!** You can test from any branch:

**Option A: Test from feature branch**
1. Push your changes (including workflow file) to your branch
2. Go to GitHub → Actions tab
3. Use the branch dropdown to select your branch
4. Find "Deploy to Hugging Face Environment" workflow
5. Click "Run workflow" → Select your branch → Run

**Option B: Merge to main first** (easier to find)
- Workflows are most visible from the default branch
- After merge, they're easier to access in the Actions tab

**Recommendation:** Test locally first, then test from your branch before merging.

### 1. Manual Workflow Dispatch

1. Go to GitHub → Actions → "Deploy to Hugging Face Environment"
2. Click "Run workflow"
3. Select:
   - Environment: `echo_env` (start with smallest)
   - Base image SHA: Leave empty (or test with specific SHA)
4. Click "Run workflow"

**What to verify:**
- ✅ Workflow runs without errors
- ✅ HF CLI installs correctly
- ✅ Script executes successfully
- ✅ Space deployed to `openenv/echo_env`

### 2. Test All Environments (Matrix)

1. Run workflow with environment: `all`
2. This will deploy all environments in parallel
3. Monitor all matrix jobs

### 3. Test Automatic Trigger

Make a change to trigger automatic deployment:

```bash
# Make a small change to core
echo "# Test" >> src/core/client_types.py
git add src/core/client_types.py
git commit -m "test: trigger deployment"
git push
```

**What to verify:**
- ✅ Workflow triggers automatically
- ✅ All environments deploy (because core changed)
- ✅ Only changed environments deploy (if only one env changed)

### 4. Check Workflow Logs

After workflow runs:
1. Click on the workflow run
2. Check each job's logs
3. Verify:
   - ✅ HF CLI installation succeeded
   - ✅ Script executed without errors
   - ✅ Deployment succeeded
   - ✅ Space URL printed at end

## Verification Checklist

### After Local Dry-Run
- [ ] Staging directory structure correct
- [ ] Dockerfile uses correct base image
- [ ] README.md has correct metadata
- [ ] All source files copied
- [ ] No sensitive files included

### After Local Deployment
- [ ] Space appears on Hugging Face
- [ ] Space builds successfully (check build logs)
- [ ] Space is accessible via web UI
- [ ] API endpoints respond correctly

### After GitHub Actions Deployment
- [ ] All matrix jobs succeeded
- [ ] Spaces deployed to correct namespace (`openenv/`)
- [ ] Spaces build successfully
- [ ] No authentication errors
- [ ] Script output shows success message

## Common Issues & Debugging

### Issue: "hf CLI not found"
```bash
# Install HF CLI locally
curl -LsSf https://hf.co/cli/install.sh | sh
# Reload shell or add to PATH
export PATH="$HOME/.local/bin:$PATH"
```

### Issue: "Authentication failed"
```bash
# Authenticate
hf auth login
# Or set token
export HF_TOKEN=your_token_here
```

### Issue: "Namespace access denied"
- Verify your HF token has access to the target namespace
- For org namespaces, token needs org permissions
- Check: `hf auth whoami`

### Issue: "Script fails in GitHub Actions"
1. Check if `HF_TOKEN` secret is set in repository settings
2. Verify token has correct permissions (write access to spaces)
3. Check workflow logs for detailed error messages

## Testing Checklist

Before merging:
- [ ] Local dry-run works for all environments
- [ ] Local deployment works to test namespace
- [ ] GitHub Actions workflow runs successfully
- [ ] At least one environment deployed via CI
- [ ] Deployed space builds and runs correctly

