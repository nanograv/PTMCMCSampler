# PTMCMCSampler Release Process

**Status**: Manual release process (requires GitHub release + conda feedstock update)

## Prerequisites
- Maintainer access to [nanograv/PTMCMCSampler](https://github.com/nanograv/PTMCMCSampler)
- Maintainer access to [conda-forge/ptmcmcsampler-feedstock](https://github.com/conda-forge/ptmcmcsampler-feedstock)
- PyPI credentials configured in GitHub secrets (`PYPI_USERNAME`, `PYPI_PASSWORD`)

## Release Steps

### 1. Prepare Release
```bash
# Ensure main branch is up to date
git checkout master
git pull origin master

# Run tests locally
make test

# Check that all CI tests are passing on GitHub
# Visit: https://github.com/nanograv/PTMCMCSampler/actions
```

### 2. Create Git Tag
```bash
# Create and push the git tag
git tag v2.1.3
git push origin v2.1.3
```

### 3. Create GitHub Release
1. Go to [PTMCMCSampler releases](https://github.com/nanograv/PTMCMCSampler/releases)
2. Click "Create a new release"
3. Choose the tag version you just created (e.g., `v2.1.3`)
4. Set release title (e.g., `v2.1.3`)
5. Add release notes describing changes
6. Click "Publish release"

### 4. Automated PyPI Upload
- GitHub Actions will automatically:
  - Run tests across Python 3.8-3.11 and Ubuntu/macOS
  - Build source distribution and wheel
  - Test the built packages (`make test-sdist`, `make test-wheel`)
  - Upload to PyPI using stored credentials

### 5. Update Conda Feedstock (Manual)
1. Go to [ptmcmcsampler-feedstock](https://github.com/conda-forge/ptmcmcsampler-feedstock)
2. Edit `recipe/meta.yaml`:
   - Update version number (line 2: `{% set version = "2.1.3" %}`)
   - Update SHA256 hash (get from PyPI package page)
3. Submit PR to conda-forge
4. Conda-forge bot will handle the rest

## Version Management
- Uses `setuptools_scm` - version automatically generated from git tags
- No manual version file updates needed
- Version is written to `PTMCMCSampler/version.py` automatically

## Troubleshooting
- **PyPI upload fails**: Check GitHub secrets are configured correctly
- **Conda build fails**: Check dependencies in `meta.yaml` are up to date
- **Tests fail**: Ensure all dependencies are installed locally before release

## Release Checklist
- [ ] All tests passing on GitHub Actions
- [ ] Release notes prepared
- [ ] Git tag created and pushed
- [ ] GitHub release created
- [ ] PyPI upload successful (check [PyPI page](https://pypi.org/project/ptmcmcsampler/))
- [ ] Conda feedstock updated
- [ ] Conda package available (check [conda-forge page](https://anaconda.org/conda-forge/ptmcmcsampler))
