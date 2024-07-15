import pathlib
import json
import os
from typing import List

from beeprint import pp  # type: ignore
from plumbum import local
import yaml
import requests
import pytest

from manager import Manager, ManagerGenerate

import config


def test_gitlab_ci_yaml_load():
    """Generates and attempts to load yaml."""
    _, rc = Manager.run(
        ["prog", "--manifest=manifests/cuda.yaml", "generate", "--ci"], exit=False
    )
    assert rc == 0
    with open(pathlib.Path(".gitlab-ci.yml")) as f:
        out = yaml.load(f, yaml.Loader)
    assert out


def test_check_moby_buildkit_set():
    """Ensures moby/buildkit version is set to the correct version in the gitlab pipeline."""
    _, rc = Manager.run(
        ["prog", "--manifest=manifests/cuda.yaml", "generate", "--ci"], exit=False
    )
    pp(config.MOBY_BUILDKIT_VERSION)
    assert rc == 0
    count = 0
    with open(pathlib.Path(".gitlab-ci.yml")) as file:
        lines = file.readlines()
        for line in lines:
            if f"image=moby/buildkit:{config.MOBY_BUILDKIT_VERSION}" in line:
                count = count + 1
    assert count == 3


def test_gitlab_ci_yaml_lint():
    """Lints the gitlab pipeline using the gitlab api."""
    y: List[str]
    with open(pathlib.Path(".gitlab-ci.yml")) as f:
        y = f.readlines()
    obj = {"content": y}
    cleans = json.dumps(obj)
    token = os.getenv("GITLAB_ACCESS_TOKEN")
    if not token:
        pytest.fail("CI_JOB_TOKEN is unset!")
    x = requests.post(
        "https://gitlab-master.nvidia.com/api/v4/ci/lint",
        data=obj,
        headers={"CI_PROJECT_ID": "12064", "PRIVATE-TOKEN": token},
    )
    pp(x)
    assert x


def test_gitlab_ci_kitmaker_arches():
    """Generates ci yaml and then checks kitmaker pipelines for correctly set arches."""
    _, rc = Manager.run(
        ["prog", "--manifest=manifests/cuda.yaml", "generate", "--ci"], exit=False
    )
    assert rc == 0
    with open(pathlib.Path(".gitlab-ci.yml")) as f:
        out = yaml.load(f, yaml.Loader)
    # pp(out)
    assert out
    assert out["kitmaker_l4t_base"]["variables"]["ARCHES"] == "tegra"
    assert all(
        x in out["kitmaker_ubuntu22.04_base"]["variables"]["ARCHES"]
        for x in ["arm64", "x86_64"]
    )
    assert all(
        x in out["kitmaker_rockylinux8_base"]["variables"]["ARCHES"]
        for x in ["arm64", "x86_64"]
    )
    assert all(
        x in out["kitmaker_ubi8_base"]["variables"]["ARCHES"]
        for x in ["arm64", "x86_64", "ppc64le"]
    )
