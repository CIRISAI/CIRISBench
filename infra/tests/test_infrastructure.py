"""
Staging Infrastructure Tests
============================

Tests for CI/CD infrastructure validation.
"""
import os
import yaml
from pathlib import Path

import pytest

# Root directory of staging
STAGING_ROOT = Path(__file__).parent.parent


class TestConfiguration:
    """Tests for configuration files."""
    
    def test_version_file_exists(self):
        """VERSION file should exist and contain valid semver."""
        version_file = STAGING_ROOT / "VERSION"
        assert version_file.exists(), "VERSION file not found"
        
        version = version_file.read_text().strip()
        parts = version.split(".")
        assert len(parts) == 3, f"Invalid semver: {version}"
        assert all(p.isdigit() for p in parts), f"Non-numeric version parts: {version}"
    
    def test_env_example_exists(self):
        """Environment example file should exist."""
        env_file = STAGING_ROOT / ".env.example"
        assert env_file.exists(), ".env.example not found"
    
    def test_feature_flags_valid(self):
        """Feature flags YAML should be valid."""
        config_file = STAGING_ROOT / "config" / "feature-flags.yaml"
        assert config_file.exists(), "feature-flags.yaml not found"
        
        config = yaml.safe_load(config_file.read_text())
        assert "flags" in config, "Missing 'flags' key"
        
        flags = config["flags"]
        # Flags can be a dict or list
        if isinstance(flags, dict):
            assert len(flags) > 0, "No flags defined"
            for name, value in flags.items():
                assert isinstance(name, str), f"Flag name must be string: {name}"
        elif isinstance(flags, list):
            assert len(flags) > 0, "No flags defined"
            for flag in flags:
                if isinstance(flag, dict):
                    assert "name" in flag, "Flag missing 'name'"
                    assert "default" in flag, f"Flag {flag.get('name')} missing 'default'"
    
    def test_models_valid(self):
        """Models YAML should be valid."""
        config_file = STAGING_ROOT / "config" / "models.yaml"
        assert config_file.exists(), "models.yaml not found"
        
        config = yaml.safe_load(config_file.read_text())
        assert "providers" in config, "Missing 'providers' key"
        assert len(config["providers"]) > 0, "No providers defined"
    
    def test_forks_valid(self):
        """Forks YAML should be valid."""
        config_file = STAGING_ROOT / "config" / "forks.yaml"
        assert config_file.exists(), "forks.yaml not found"
        
        config = yaml.safe_load(config_file.read_text())
        assert "repositories" in config, "Missing 'repositories' key"
        
        repos = config["repositories"]
        assert "ethicsengine" in repos, "Missing ethicsengine repo"
        assert "cirisnode" in repos, "Missing cirisnode repo"
    
    def test_environment_configs_valid(self):
        """Environment config files should be valid."""
        envs = ["development", "staging", "production"]
        
        for env in envs:
            config_file = STAGING_ROOT / "config" / "environments" / f"{env}.yaml"
            assert config_file.exists(), f"{env}.yaml not found"
            
            config = yaml.safe_load(config_file.read_text())
            assert "environment" in config, f"{env}.yaml missing 'environment' key"
            assert config["environment"] == env, f"Environment mismatch in {env}.yaml"


class TestDockerCompose:
    """Tests for Docker Compose files."""
    
    @pytest.fixture
    def docker_dir(self):
        return STAGING_ROOT / "docker"
    
    def test_compose_files_exist(self, docker_dir):
        """Required Docker Compose files should exist."""
        required = [
            "docker-compose.he300.yml",
            "docker-compose.dev.yml",
            "docker-compose.test.yml",
            "docker-compose.prod.yml",
        ]
        
        for filename in required:
            filepath = docker_dir / filename
            assert filepath.exists(), f"{filename} not found"
    
    def test_compose_files_valid_yaml(self, docker_dir):
        """Docker Compose files should be valid YAML."""
        for filepath in docker_dir.glob("*.yml"):
            config = yaml.safe_load(filepath.read_text())
            assert "services" in config, f"{filepath.name} missing 'services' key"
    
    def test_he300_has_required_services(self, docker_dir):
        """HE-300 compose should have all required services."""
        config = yaml.safe_load((docker_dir / "docker-compose.he300.yml").read_text())
        
        # Check for required services (db = postgres alias)
        required_services = ["cirisnode", "eee", "redis"]
        for service in required_services:
            assert service in config["services"], f"Missing service: {service}"
        
        # postgres can be named 'postgres' or 'db'
        assert "postgres" in config["services"] or "db" in config["services"], \
            "Missing postgres/db service"


class TestScripts:
    """Tests for shell scripts."""
    
    @pytest.fixture
    def scripts_dir(self):
        return STAGING_ROOT / "scripts"
    
    def test_scripts_exist(self, scripts_dir):
        """Required scripts should exist."""
        required = [
            "setup-submodules.sh",
            "run-tests.sh",
            "run-benchmark.sh",
            "version-bump.sh",
            "build-images.sh",
            "sync-forks.sh",
            "deploy.sh",
        ]
        
        for script in required:
            filepath = scripts_dir / script
            assert filepath.exists(), f"{script} not found"
    
    def test_scripts_have_shebang(self, scripts_dir):
        """Scripts should have bash shebang."""
        for filepath in scripts_dir.glob("*.sh"):
            content = filepath.read_text()
            assert content.startswith("#!/"), f"{filepath.name} missing shebang"
            assert "bash" in content.split("\n")[0], f"{filepath.name} not bash script"


class TestWorkflows:
    """Tests for GitHub Actions workflows."""
    
    @pytest.fixture
    def workflows_dir(self):
        return STAGING_ROOT / ".github" / "workflows"
    
    def test_workflows_exist(self, workflows_dir):
        """Required workflows should exist."""
        required = [
            "ci.yml",
            "he300-benchmark.yml",
            "regression.yml",
            "release.yml",
            "benchmark.yml",
            "submodule-sync.yml",
        ]
        
        for workflow in required:
            filepath = workflows_dir / workflow
            assert filepath.exists(), f"{workflow} not found"
    
    def test_workflows_valid_yaml(self, workflows_dir):
        """Workflows should be valid YAML."""
        for filepath in workflows_dir.glob("*.yml"):
            config = yaml.safe_load(filepath.read_text())
            assert "name" in config, f"{filepath.name} missing 'name'"
            # 'on' in YAML is parsed as boolean True, so check for both
            assert "on" in config or True in config, f"{filepath.name} missing 'on' trigger"
            assert "jobs" in config, f"{filepath.name} missing 'jobs'"


class TestMakefile:
    """Tests for Makefile."""
    
    def test_makefile_exists(self):
        """Makefile should exist."""
        makefile = STAGING_ROOT / "Makefile"
        assert makefile.exists(), "Makefile not found"
    
    def test_makefile_has_required_targets(self):
        """Makefile should have required targets."""
        makefile = STAGING_ROOT / "Makefile"
        content = makefile.read_text()
        
        required_targets = [
            "help",
            "setup",
            "test",
            "dev-up",
            "dev-down",
            "benchmark",
        ]
        
        for target in required_targets:
            assert f"{target}:" in content, f"Missing target: {target}"


class TestJenkinsfile:
    """Tests for Jenkinsfile."""
    
    def test_jenkinsfile_exists(self):
        """Jenkinsfile should exist."""
        jenkinsfile = STAGING_ROOT / "Jenkinsfile"
        assert jenkinsfile.exists(), "Jenkinsfile not found"
    
    def test_jenkinsfile_has_pipeline(self):
        """Jenkinsfile should define a pipeline."""
        jenkinsfile = STAGING_ROOT / "Jenkinsfile"
        content = jenkinsfile.read_text()
        
        assert "pipeline" in content, "Missing pipeline block"
        assert "stages" in content, "Missing stages block"
